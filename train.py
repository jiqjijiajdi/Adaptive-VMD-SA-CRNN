# =============================================================================
# Copyright (c) 2025 Yaolin Zou, Xiongfei Tao, Kenan Xiao, Jinkang Shi.
# All rights reserved.
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import argparse

# Local imports
from model import EndToEndNet
from loss import DiceBCELoss, SafeZoneLoss
from preprocess import EndToEndDataset, FixedFileDataset, generate_validation_file

def plot_training_curves(loss_history, acc_history):
    epochs = range(1, len(acc_history) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', color=color, fontsize=12)
    l1, = ax1.plot(epochs, loss_history['total'], color=color, label='Total Loss', linestyle='-', linewidth=2)
    l2, = ax1.plot(epochs, loss_history['seg'], color='orange', label='Seg Loss', linestyle='--', alpha=0.7)
    l3, = ax1.plot(epochs, loss_history['count'], color='salmon', label='Count Loss', linestyle=':', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Safe Accuracy (%)', color=color, fontsize=12)  
    l4, = ax2.plot(epochs, acc_history, color=color, label='Validation Acc', linewidth=2.5, marker='o', markersize=4)
    ax2.tick_params(axis='y', labelcolor=color)
    
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=10)
    
    plt.title('Training Convergence & Performance Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Figure4_Training_Curves.pdf', dpi=600)
    print("✅ Figure 4 saved as 'Figure4_Training_Curves.pdf'")
    # plt.show() # Uncomment if you have a display

def train_end_to_end():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting End-to-End Training on {device}...")

    # Initialize Model
    model = EndToEndNet().to(device)
    if os.path.exists("resunet_best.pth"):
        print(" -> Loading pre-trained ResUNet weights...")
        try:
            model.seg_net.load_state_dict(torch.load("resunet_best.pth", map_location=device, weights_only=True))
        except Exception as e:
            print(f"Warning: Could not load ResUNet weights: {e}")

    # Optimizer & Scheduler
    optimizer = optim.AdamW([
        {'params': model.seg_net.parameters(), 'lr': 1e-4}, 
        {'params': model.count_net.parameters(), 'lr': 1e-4} 
    ], weight_decay=1e-4)
    
    epochs = 80
    batch_size = 32
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[1e-4, 1e-4], steps_per_epoch=5000//batch_size, epochs=epochs)
    
    # Losses
    criterion_seg = DiceBCELoss().to(device)
    criterion_count = SafeZoneLoss().to(device)
    
    # Datasets
    # 1. Training Set (Dynamic generation)
    train_ds = EndToEndDataset(epoch_size=5000, augment=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, drop_last=True)
    
    # 2. Validation Set (Fixed file)
    val_file = "validation_dataset.pt"
    if not os.path.exists(val_file):
        generate_validation_file(val_file, num_samples=2000)
        
    test_ds = FixedFileDataset(val_file)
    test_loader = DataLoader(test_ds, batch_size=batch_size, drop_last=False)
    
    idx_tensor = torch.arange(65).to(device).float()
    
    loss_history = {'total': [], 'seg': [], 'count': []}
    acc_history = []
    
    best_acc = 0.0
    patience, counter = 20, 0
    
    for epoch in range(epochs):
        model.train()
        l_seg_sum, l_count_sum, l_total_sum = 0, 0, 0
        
        for spec, mask_gt, ldl_gt, k_real in train_loader:
            spec, mask_gt, ldl_gt, k_real = spec.to(device), mask_gt.to(device), ldl_gt.to(device), k_real.to(device)
            
            optimizer.zero_grad()
            seg_logits, count_logits = model(spec)
            
            loss_seg = criterion_seg(seg_logits, mask_gt)
            loss_count = criterion_count(count_logits, ldl_gt, k_real)
            loss_total = 1.0 * loss_seg + 1.0 * loss_count
            
            loss_total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            l_seg_sum += loss_seg.item()
            l_count_sum += loss_count.item()
            l_total_sum += loss_total.item()
            
        avg_total = l_total_sum / len(train_loader)
        avg_seg = l_seg_sum / len(train_loader)
        avg_count = l_count_sum / len(train_loader)
        
        loss_history['total'].append(avg_total)
        loss_history['seg'].append(avg_seg)
        loss_history['count'].append(avg_count)
        
        print(f"Ep {epoch+1}/{epochs}: Loss={avg_total:.3f} (Seg={avg_seg:.3f}, Count={avg_count:.3f})")
        
        # Validation
        model.eval()
        safe_acc, total = 0, 0
        with torch.no_grad():
            for spec, _, _, k_real in test_loader:
                spec, k_real = spec.to(device), k_real.to(device)
                _, count_logits = model(spec)
                
                # Prediction Logic
                probs = F.softmax(count_logits, dim=1)
                expectation = torch.sum(probs * idx_tensor, dim=1)
                k_pred = torch.round(expectation)
                
                # Safe Accuracy: [k_real, k_real + 2]
                safe_acc += ((k_pred >= k_real) & (k_pred <= k_real + 2)).sum().item()
                total += spec.size(0)
        
        curr_acc = safe_acc / total * 100
        acc_history.append(curr_acc)
        print(f"  --> Acc: {curr_acc:.2f}% (Best: {best_acc:.2f}%)")
        
        if curr_acc > best_acc:
            best_acc = curr_acc
            counter = 0
            torch.save(model.state_dict(), "end2end_best.pth")
            print("  --> Model Saved!")
        else:
            counter += 1
            if counter >= patience:
                print("🛑 Early Stopping!")
                break
    
    plot_training_curves(loss_history, acc_history)

if __name__ == "__main__":
    train_end_to_end()