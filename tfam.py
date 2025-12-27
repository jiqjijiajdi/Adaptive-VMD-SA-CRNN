# =============================================================================
# Copyright (c) 2025 Yaolin Zou, Xiongfei Tao, Kenan Xiao, Jinkang Shi.
# All rights reserved.
# =============================================================================

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), 
            nn.BatchNorm2d(out_c), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(out_c, out_c, 3, padding=1), 
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential()
        if in_c != out_c: 
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1), nn.BatchNorm2d(out_c))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x): 
        return self.relu(self.conv(x) + self.shortcut(x))

class ResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = ResidualBlock(1, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d((2,1)), ResidualBlock(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d((2,1)), ResidualBlock(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d((2,1)), ResidualBlock(128, 256))
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=(2,1), stride=(2,1))
        self.conv1 = ResidualBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=(2,1), stride=(2,1))
        self.conv2 = ResidualBlock(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=(2,1), stride=(2,1))
        self.conv3 = ResidualBlock(64, 32)
        
        self.outc = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        
        return self.outc(x)