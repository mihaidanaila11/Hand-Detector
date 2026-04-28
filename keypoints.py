import torch.nn as nn
import torch
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = self.relu(x)
        return x
    
class MSAB(nn.Module):
    def __init__(self, channels):
        super(MSAB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels//4, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels//4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels, channels//4, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Linear(channels, channels//8)
        
        self.fc1 = nn.Linear(channels//8, self.c_prime)
        self.fc2 = nn.Linear(channels//8, self.c_prime)
        self.fc3 = nn.Linear(channels//8, self.c_prime)
        self.fc4 = nn.Linear(channels//8, self.c_prime)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        
        x_a = out1 + out2 + out3 + out4
        f = self.avg_pool(x_a)
        f = f.view(f.size(0), -1)
        f = self.fc(f)
        w1  = self.fc1(f)
        w2  = self.fc2(f)
        w3  = self.fc3(f)
        w4  = self.fc4(f)
        
        weights = torch.stack((w1, w2, w3, w4), dim=1)
        weights = F.softmax(weights, dim=1)
        
        x_c = torch.stack((out1, out2, out3, out4), dim=1)
        
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        out = x_c * weights
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # input 3x256x256
        self.layer1 = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 32)
        )
        self.maxpool1 = nn.MaxPool2d(2, 2)
        
        self.layer2 = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 64)
        )
        self.maxpool2 = nn.MaxPool2d(2, 2)
        
        self.layer3 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128)
        )


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x