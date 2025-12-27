# =============================================================================
# Copyright (c) 2025 Yaolin Zou, Xiongfei Tao, Kenan Xiao, Jinkang Shi.
# All rights reserved.
# =============================================================================

import torch
import torch.nn as nn
from tfam import ResUNet

class CRNNCounter(nn.Module):
    def __init__(self, num_classes=65):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 1)), 
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 1)), 
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 1)), 
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1))
        )
        self.rnn = nn.GRU(input_size=256*16, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.4)
        self.attn1 = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.cnn(x)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, -1) 
        out, _ = self.rnn(x)
        attn_out1, _ = self.attn1(out, out, out)
        attn_out2, _ = self.attn2(attn_out1, attn_out1, attn_out1)
        x = torch.mean(attn_out2, dim=1)
        return self.fc(x)

class EndToEndNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seg_net = ResUNet()
        self.count_net = CRNNCounter()

    def forward(self, x):
        seg_logits = self.seg_net(x)
        soft_mask = torch.sigmoid(seg_logits)
        masked_spec = x * soft_mask
        count_logits = self.count_net(masked_spec)
        return seg_logits, count_logits