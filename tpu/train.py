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
import wandb

try:
    import torch_xla.core.xla_model as xm

    _xla_available = True
except:
    _xla_available = False


def train_model_tpu():
    train = datasets.ImageFolder(cfg.TRAIN_DIR, transform=cfg.train_transform)
    valid = datasets.ImageFolder(cfg.VAL_DIR, transform=cfg.train_transform)

    torch.manual_seed(42)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True
    )

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=cfg.train_bs,
        sampler=train_sampler,
        num_workers=0,
        drop_last=True,
    )

    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        train, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True
    )

    valid_loader = torch.utils.data.DataLoader(
        train,
        batch_size=cfg.valid_bs,
        sampler=valid_sampler,
        num_workers=0,
        drop_last=True,
    )

    xm.master_print(f"Train for {len(train_loader)} steps per epoch")

    # Scale learning rate to num cores
    learning_rate = 0.0001 * xm.xrt_world_size()

    # Get loss function, optimizer, and model
    device = xm.xla_device()

    model = Model()

    for param in model.base_model.parameters():
        param.requires_grad = False

    # Log gradients and model params to wandb
    # wandb.watch(model)

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        div_factor=10.0,
        final_div_factor=50.0,
        epochs=cfg.epochs,
        steps_per_epoch=len(train_loader),
    )

    accuracy_stats = {"train": [], "val": []}

    loss_stats = {"train": [], "val": []}

    def train_loop_fn(loader):
        tracker = xm.RateTracker()
        model.train()
        total_samples_train, correct_train = 0, 0

        # Training and calculating train accuracy and loss
        for x, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            output = model(data)
            train_loss = loss_fn(output, target)
            train_loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(data.shape[0])

            pred_train = output.max(1, keepdim=True)[1]
            correct_train += pred_train.eq(target.view_as(pred_train)).sum().item()
            total_samples_train += data.size()[0]

            scheduler.step()
            if x % 40 == 0:
                print(
                    "[xla:{}]({})\tLoss={:.3f}\tRate={:.2f}\tGlobalRate={:.2f}".format(
                        xm.get_ordinal(),
                        x,
                        train_loss.item(),
                        tracker.rate(),
                        tracker.global_rate(),
                    ),
                    flush=True,
                )

        train_accuracy = 100.0 * correct_train / total_samples_train
        print(
            "[xla:{}] Accuracy={:.2f}%".format(xm.get_ordinal(), train_accuracy),
            flush=True,
        )
        return train_accuracy

    # Train loops
    train_accuracy = []
    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        para_loader = pl.ParallelLoader(train_loader, [device])
        train_accuracy.append(train_loop_fn(para_loader.per_device_loader(device)))
        xm.master_print(
            "Finished training epoch {} train-acc {:.2f} in {:.2f} sec".format(
                epoch, train_accuracy[-1], (time.time() - start)
            )
        )
        # xm.save(model.state_dict(), "./model.pt")

        if epoch == 15:  # unfreeze
            for param in model.base_model.parameters():
                param.requires_grad = True

    return train_accuracy


if __name__ == "__main__":

    import time

    # wandb.init(project="GPU-vs-TPU")
    start_time = time.time()
    # config = wandb.config
    # config.learning_rate = cfg.lr
    # config.epochs = cfg.epochs
    # config.train_batch_size = cfg.train_bs
    # config.valid_batch_size = cfg.valid_bs

    def _mp_fn(rank, flags):
        global acc_list
        torch.set_default_tensor_type("torch.FloatTensor")
        a = train_model_tpu()

    FLAGS = {}
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method="fork")

    end_time = time.time() - start_time
    # wandb.log({"Time_taken": end_time})
