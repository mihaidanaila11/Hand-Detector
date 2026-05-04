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

        self.c_prime = channels // 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, self.c_prime, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.c_prime),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, self.c_prime, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.c_prime),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, self.c_prime, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.c_prime),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels, self.c_prime, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(self.c_prime),
            nn.ReLU(inplace=True)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Linear(self.c_prime, channels//8)
        self.reluFC = nn.ReLU(inplace=True)

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
        f = self.reluFC(f)
        w1  = self.fc1(f)
        w2  = self.fc2(f)
        w3  = self.fc3(f)
        w4  = self.fc4(f)
        
        weights = torch.stack((w1, w2, w3, w4), dim=1)
        weights = F.softmax(weights, dim=1)
        
        x_c = torch.stack((out1, out2, out3, out4), dim=1)
        
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        out = x_c * weights
        return out.reshape(x.size(0), x.size(1), x.size(2), x.size(3))

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

        self.masb3 = MSAB(128)

        self.maxpool3 = nn.MaxPool2d(2, 2)

        self.layer4 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256)
        )

        self.masb4 = MSAB(256)


    def forward(self, x):
        # output 32x256x256
        c1 = self.layer1(x)
        p1 = self.maxpool1(c1)

        # output 64x128x128
        c2 = self.layer2(p1)
        p2 = self.maxpool2(c2)

        # output 128x64x64
        c3 = self.layer3(p2)
        msab3 = self.masb3(c3)
        p3 = self.maxpool3(msab3)

        # output 256x32x32
        c4 = self.layer4(p3)
        msab4 = self.masb4(c4)
        return msab4, c1, c2, c3

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(
         
         ConvBlock(384, 128),
         ConvBlock(128, 128)        
        )

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Sequential(
         ConvBlock(192, 64),
         ConvBlock(64, 64)        
        )

        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Sequential(
         ConvBlock(96, 32),
         ConvBlock(32, 32)        
        )

        self.finalCConv = nn.Sequential(
            nn.Conv2d(32, 21, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x, enc_1, enc_2, enc_3):
        # enc 1 - 32x256x256, 
        # enc 2 - 64x128x128, 
        # enc 3 - 128x64x64

        # up1 - 256x64x64
        # up2 - 128x128x128
        # up3 - 64x256x256

        # cat1 - 256+128 = 384x64x64
        # cat2 - 128+64 = 192x128x128
        # cat3 - 64+32 = 96x256x256

        # output 256x64x64
        x = self.upsample1(x)
        x = torch.cat((x, enc_3), dim=1)
        x = self.conv1(x)

        x = self.upsample2(x)
        x = torch.cat((x, enc_2), dim=1)
        x = self.conv2(x)

        x = self.upsample3(x)
        x = torch.cat((x, enc_1), dim=1)
        x = self.conv3(x)
        x = self.finalCConv(x)
        return x

class HandKeypointDetector(nn.Module):
    def __init__(self):
        super(HandKeypointDetector, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x, enc_1, enc_2, enc_3 = self.encoder(x)
        x = self.decoder(x, enc_1, enc_2, enc_3)
        return x

import torch
import torch.nn as nn

class IoULossHeatmap(nn.Module):
    def __init__(self, num_keypoints=21, sigma=3):
        super(IoULossHeatmap, self).__init__()
        self.K = num_keypoints
        self.sigma = sigma

    def generate_gaussian_heatmap(self, size, keypoint, device):
        """
        Generează un heatmap Gaussian 2D pentru un singur punct.
        """
        h, w = size
        u, v = keypoint
        
        # Creăm grila de coordonate direct pe device-ul corect (GPU/CPU)
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        grid_y = grid_y.to(device)
        grid_x = grid_x.to(device)
        
        # Formula Gaussiană 2D[cite: 3]
        gaussian = torch.exp(-((grid_x - u)**2 + (grid_y - v)**2) / (2 * self.sigma**2))
        
        # Normalizăm astfel încât vârful să fie exact 1.0
        return gaussian / (gaussian.max() + 1e-7)  

    def forward(self, h_pred, labels):
        """
        h_pred: Tensor de forma (Batch, K, H, W) - Ieșirea modelului tău
        labels: Lista de tensori 1D (fără bbox), care vin din DataLoader
        """
        h_pred = h_pred.float()
        batch_size = h_pred.size(0)
        # h, w = h_pred.size(2), h_pred.size(3)
        h,w = 128, 128
        device = h_pred.device

        if h_pred.size(2) != h or h_pred.size(3) != w:
            h_pred = torch.nn.functional.interpolate(
                h_pred, 
                size=(h, w), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Inițializăm tensorul ground-truth cu zerouri, de aceeași formă cu predicția
        h_true = torch.zeros((batch_size, self.K, h, w), device=device, dtype=h_pred.dtype)

        grid_y = torch.arange(h, device=device, dtype=h_pred.dtype).view(1, h, 1)
        grid_x = torch.arange(w, device=device, dtype=h_pred.dtype).view(1, 1, w)
        
        for b in range(batch_size):
            label = labels[b]
            
            # Fără bbox, label-ul conține DOAR cele 21 de puncte.
            # -1 permite PyTorch să deducă automat dacă e (21, 3) sau (21, 2)
            kps = label.view(self.K, -1) 

            keypoints = kps[:, :2]
            x_px = keypoints[:, 0].unsqueeze(1).unsqueeze(2) * w
            y_px = keypoints[:, 1].unsqueeze(1).unsqueeze(2) * h

            gaussian = torch.exp(-((grid_x - x_px) ** 2 + (grid_y - y_px) ** 2) / (2 * self.sigma**2))
            h_true[b] = gaussian / (gaussian.amax(dim=(1, 2), keepdim=True) + 1e-7)
            
        # ---------------- Calculul Modified IoU[cite: 3] ---------------- #
        
        # Intersecția: Produsul element cu element
        intersection = torch.sum(h_pred * h_true, dim=(2, 3))
        
        # Pătratele pentru a calcula reuniunea
        pred_sq = torch.sum(h_pred ** 2, dim=(2, 3))
        true_sq = torch.sum(h_true ** 2, dim=(2, 3))
        
        # Reuniunea
        union = pred_sq + true_sq - intersection
        
        # Calculul IoU per fiecare keypoint, cu un epsilon pentru a evita divizarea la zero
        iou_per_keypoint = (intersection + 1e-7) / (union + 1e-7)
        
        # Sumăm IoU-urile celor 21 de puncte pentru fiecare imagine din batch
        sum_iou = torch.sum(iou_per_keypoint, dim=1) 
        
        # Loss-ul final este 1 minus media aritmetică a IoU-urilor[cite: 3]
        loss_final = 1 - (torch.mean(sum_iou) / self.K)
        
        return loss_final