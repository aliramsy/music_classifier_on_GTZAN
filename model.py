import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18MusicGenre(nn.Module):
    def __init__(self, num_classes=10, input_channels=1, drop_out_rate=.0005):  
        super(ResNet18MusicGenre, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.fc = nn.Sequential(
            nn.Dropout(drop_out_rate),  
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )
        #self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

num_genres = 11  
model = ResNet18MusicGenre(num_classes=num_genres, input_channels=1)

if __name__ == '__main__':
    print(model)
