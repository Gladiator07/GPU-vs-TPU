import time
from dataset import ClassificationDataset, load_pickle_file
import config as cfg
import torch
from model import Model
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

try:
    import torch_xla.core.xla_model as xm
    
    _xla_available = True
except:
    _xla_available = False


def train_model():
    train_ids = load_pickle_file(cfg.train_ids_224_pkl)
    train_class = load_pickle_file(cfg.train_class_224_pkl)
    train_images = load_pickle_file(cfg.train_image_224_pkl)

    val_ids = load_pickle_file(cfg.val_ids_224_pkl)
    val_class = load_pickle_file(cfg.val_class_224_pkl)
    val_images = load_pickle_file(cfg.val_image_224_pkl)
    
    train_dataset = ClassificationDataset(id=train_ids, classes = train_class, images = train_images)
    val_dataset = ClassificationDataset(id=val_ids, classes=val_class, images = val_images, is_valid=True)

    torch.manual_seed(42)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas = xm.xrt_world_size(),
        rank = xm.get_ordinal(),
        shuffle = True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = cfg.train_bs,
        sampler = train_sampler,
        num_workers = 0,
        drop_last = True
    )

    xm.master_print(f"Train for {len(train_loader)} steps per epoch")

    # Scale learning rate to num cores
    learning_rate = cfg.lr * xm.xrt_world_size()

    device = xm.xla_device()


    model = Model()
    for param in model.base_model.parameters():  # freeze some layers
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