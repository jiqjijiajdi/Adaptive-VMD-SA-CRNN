# =============================================================================
# Copyright (c) 2025 Yaolin Zou, Xiongfei Tao, Kenan Xiao, Jinkang Shi.
# All rights reserved.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3))
        dice = 1 - (2. * intersection + self.smooth) / (union + self.smooth)
        return self.bce(logits, targets) + dice.mean()

class SafeZoneLoss(nn.Module):
    def __init__(self, num_classes=65):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.huber = nn.SmoothL1Loss()
        # Ensure idx_tensor is moved to the correct device in forward pass or init
        self.idx_tensor = torch.arange(num_classes).float()

    def forward(self, logits, target_dist, target_k):
        if self.idx_tensor.device != logits.device:
            self.idx_tensor = self.idx_tensor.to(logits.device)
            
        loss_kl = self.kl(F.log_softmax(logits, dim=1), target_dist)
        probs = F.softmax(logits, dim=1)
        expectation = torch.sum(probs * self.idx_tensor, dim=1)
        loss_reg = self.huber(expectation, target_k.float())
        
        diff_under = F.relu(target_k - expectation) 
        diff_over = F.relu(expectation - (target_k + 2.2)) 
        loss_constraint = 5 * (diff_under ** 2) + 0.5 * (diff_over ** 2)
        
        return loss_kl + loss_reg + 0.1 * loss_constraint.mean()