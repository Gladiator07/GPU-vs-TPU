import time
import config as cfg
import torch
from model import Model
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

try:
    import torch_xla.core.xla_model as xm
    
    _xla_available = True
except:
    _xla_available = False


def train_model():
    train = datasets.ImageFolder(cfg.TRAIN_DIR, transform=cfg.train_transform)
    valid = datasets.ImageFolder(cfg.VAL_DIR, transform=cfg.train_transform)

    train = torch.utils.data.ConcatDataset([train, valid])
    
    torch.manual_seed(42)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
    
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=cfg.train_bs,
        sampler=train_sampler,
        num_workers=0,
        drop_last=True) # print(len(train_loader))
    
    
    xm.master_print(f"Train for {len(train_loader)} steps per epoch")
    
    # Scale learning rate to num cores
    learning_rate = 0.0001 * xm.xrt_world_size()

    # Get loss function, optimizer, and model
    device = xm.xla_device()

    model = Model()
    
    for param in model.base_model.parameters(): # freeze some layers
        param.requires_grad = False
    
    model = model.to(device)
    loss_fn =  nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = OneCycleLR(optimizer, 
                           learning_rate, 
                           div_factor=10.0, 
                           final_div_factor=50.0, 
                           epochs=cfg.epochs,
                           steps_per_epoch=len(train_loader))
    
    
    
    def train_loop_fn(loader):
        tracker = xm.RateTracker()
        model.train()
        total_samples, correct = 0, 0
        for x, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(data.shape[0])
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size()[0]
            scheduler.step()
            if x % 40 == 0:
                print('[xla:{}]({})\tLoss={:.3f}\tRate={:.2f}\tGlobalRate={:.2f}'.format(
                    xm.get_ordinal(), x, loss.item(), tracker.rate(),
                    tracker.global_rate()), flush=True)
        accuracy = 100.0 * correct / total_samples
        print('[xla:{}] Accuracy={:.2f}%'.format(xm.get_ordinal(), accuracy), flush=True)
        return accuracy

    # Train loops
    accuracy = []
    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        para_loader = pl.ParallelLoader(train_loader, [device])
        accuracy.append(train_loop_fn(para_loader.per_device_loader(device)))
        xm.master_print("Finished training epoch {} train-acc {:.2f} in {:.2f} sec"\
                        .format(epoch, accuracy[-1], time.time() - start))
        xm.save(model.state_dict(), "./model.pt")

#         if epoch == 15: #unfreeze
#             for param in model.base_model.parameters():
#                 param.requires_grad = True

    return accuracy

if __name__ == "__main__":
    
    # Start training processes
    def _mp_fn(rank, flags):
        global acc_list
        torch.set_default_tensor_type('torch.FloatTensor')
        a = train_model()

    FLAGS={}
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')