import config as cfg
import torch
from model import Model
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets
from tqdm import tqdm
import wandb


def accuracy(output, target):
    y_pred_softmax = torch.log_softmax(output, dim=1)
    _, preds = torch.max(y_pred_softmax, dim=1)
    correct_pred = (preds == target).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = acc * 100
    return acc


def train_model_gpu():

    train_data = datasets.ImageFolder(cfg.TRAIN_DIR, transform=cfg.train_transform)
    valid_data = datasets.ImageFolder(cfg.VAL_DIR, transform=cfg.train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.train_bs, shuffle=True, num_workers=2
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=cfg.valid_bs, shuffle=False, num_workers=2
    )

    epochs = cfg.epochs
    batch_size = cfg.train_bs
    learning_rate = cfg.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training model on {device}...")

    model = Model().to(device)

    for param in model.base_model.parameters():  # freeze some layers
        param.requires_grad = False

    # Log gradients and model params to wandb
    wandb.watch(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        div_factor=10.0,
        final_div_factor=50.0,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    accuracy_stats = {"train": [], "val": []}

    loss_stats = {"train": [], "val": []}

    print("Starting training...")

    for epoch in range(1, epochs + 1):

        train_epoch_loss = 0
        train_epoch_acc = 0

        model.train()

        with tqdm(train_loader, unit="batch") as tepoch:
            for image, target in tepoch:
                image = image.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(image)
                train_loss = criterion(output, target)
                train_acc = accuracy(output, target)
                train_loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Log to wandb
                wandb.log({"train_loss": train_loss})
                wandb.log({"train_accuracy": train_acc})

                tepoch.set_description(f"Epoch [{epoch}/{epochs}")
                tepoch.set_postfix(loss=train_loss.item(), accuracy=train_acc.item())

                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()

        # validate model
        with torch.no_grad():
            model.eval()
            val_epoch_loss = 0
            val_epoch_acc = 0
            for image, target in valid_loader:
                image = image.to(device)
                target = target.to(device)

                output_val = model(image)
                val_loss = criterion(output_val, target)
                val_acc = accuracy(output_val, target)
                
                # Log to wandb
                wandb.log({"valid_loss": val_loss})
                wandb.log({"valid_accuracy": val_acc})

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        loss_stats["train"].append(train_epoch_loss / len(train_loader))
        loss_stats["val"].append(val_epoch_loss / len(valid_loader))
        accuracy_stats["train"].append(train_epoch_acc / len(train_loader))
        accuracy_stats["val"].append(val_epoch_acc / len(valid_loader))
        print(
            f"Epoch {epoch+0:02}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(valid_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(valid_loader):.3f}"
        )


if __name__ == "__main__":
    
    wandb.init(project="GPU-vs-TPU")
    
    config = wandb.config
    config.learning_rate = cfg.lr
    config.epochs = cfg.epochs
    config.train_batch_size = cfg.train_bs
    config.valid_batch_size = cfg.valid_bs

    train_model_gpu()
