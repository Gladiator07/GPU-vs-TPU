
import torch.nn as nn
import efficientnet_pytorch
import torchvision
import torch
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        
        self.base_model = torchvision.models.densenet201(pretrained=True)
        self.base_model.classifier = nn.Identity()
        self.fc = torch.nn.Sequential(
                    torch.nn.Linear(1920, 1024, bias = True),
                    torch.nn.BatchNorm1d(1024),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(1024, 512, bias = True),
                    torch.nn.BatchNorm1d(512),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(512, 104))
        
    def forward(self, image, targets):
        # x = self.base_model(inputs)
        # return self.fc(x)
        out = self.base_model(image)
        out = self.fc(out)
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(out))
        return out, loss

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.base_model = efficientnet_pytorch.EfficientNet.from_pretrained(
#             'efficientnet-b0'
#         )
#         self.base_model._fc = nn.Linear(
#             in_features=1280, 
#             out_features=1, 
#             bias=True
#         )
        
#     def forward(self, image, targets):
#         print(image.shape)
#         out = self.base_model(image)
#         loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(out))
#         return out, loss
