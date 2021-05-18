import config as cfg
import torch
from model import Model
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets


# for calculating the accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
def train_model_gpu():

    train_data = datasets.ImageFolder(cfg.TRAIN_DIR, transform=cfg.train_transform)
    valid_data = datasets.ImageFolder(cfg.VAL_DIR, transform=cfg.train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size = cfg.train_bs,
        shuffle=True,
        num_workers = 2
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_data, 
        batch_size = cfg.valid_bs,
        shuffle=False,
        num_workers = 2
    )

    epochs = cfg.epochs
    batch_size = cfg.train_bs
    learning_rate = cfg.lr

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Training model on cuda...")
    
    model = Model().to(device)
    
    for param in model.base_model.parameters(): # freeze some layers
        param.requires_grad = False
    
    criterion =  nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = OneCycleLR(optimizer, 
                           learning_rate, 
                           div_factor=10.0, 
                           final_div_factor=50.0, 
                           epochs=epochs,
                           steps_per_epoch=len(train_loader))
    
    train_losses = []
    valid_losses = []

    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()

        correct = 0
        for image, target in train_loader:
            image = image.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            
            train_loss += loss.item() * image.size(0)

            print(f"{torch.max(output, dim=1)}")
            print(f"{torch.max(output, dim=1) == target}")
            
            correct += (torch.max(output, dim=1) == target).float().sum()

        accuracy = 100 * correct / len(train_data)
        # validate model
        model.eval()
        for image, target in valid_loader:
            image  = image.to(device)
            target = target.to(device)

            output = model(image)
            loss = criterion(output, target)

            # update validation loss

            valid_loss += loss.item() * image.size(0)

        # calculate average losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sample)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

    # print-training/validation-statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\t Accuracy: {:.6f}'.format(
        epoch, train_loss, valid_loss, accuracy))
    


if __name__ == "__main__":

    train_model_gpu()