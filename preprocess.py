# =============================================================================
# Copyright (c) 2025 Yaolin Zou, Xiongfei Tao, Kenan Xiao, Jinkang Shi.
# All rights reserved.
# =============================================================================

import torch
import numpy as np
from scipy import signal
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
import os
from simulation import SignalSimulator, TimeDomainAugmentor

class EndToEndDataset(IterableDataset):
    def __init__(self, epoch_size=5000, augment=True):
        self.sim = SignalSimulator(T=0.4) 
        self.epoch_size = epoch_size
        self.df = self.sim.fs / 1024
        self.augment = augment
        self.augmentor = TimeDomainAugmentor(p_gain=0.6, p_noise=0.3, p_cutout=0.0)
        self.nperseg = 1024
        self.noverlap = 960
        self.target_T = 64 

    def __iter__(self):
        for _ in range(self.epoch_size):
            sig, K_real, tracks = self.sim.generate()
            if self.augment: 
                sig = self.augmentor(sig)
            
            # STFT
            f, t_stft, Zxx = signal.stft(sig, fs=self.sim.fs, nperseg=self.nperseg, noverlap=self.noverlap, nfft=1024)
            spec = np.abs(Zxx)[:256, :] 
            
            # Preprocessing & Resizing
            spec_tensor = torch.tensor(spec).unsqueeze(0).unsqueeze(0).float() 
            spec_resized = F.interpolate(spec_tensor, size=(256, self.target_T), mode='bilinear', align_corners=False).squeeze(0) 
            
            # Normalization
            spec_val = spec_resized[0].numpy()
            spec_min, spec_max = spec_val.min(), spec_val.max()
            spec_resized = (spec_resized - spec_min) / (spec_max - spec_min + 1e-8)
            
            # Generate Attention Mask
            mask = np.zeros((256, self.target_T), dtype=np.float32)
            y_coords = np.arange(256)
            resampled_indices = np.linspace(0, self.sim.N - 1, self.target_T).astype(int)
            
            for track in tracks:
                freqs_curr = track[resampled_indices]
                for ti, f_val in enumerate(freqs_curr):
                    center = f_val / self.df
                    if 0 <= center < 256:
                        low, high = max(0, int(center)-3), min(256, int(center)+4)
                        vals = np.exp(-0.5 * ((y_coords[low:high] - center) / 1.0)**2)
                        mask[low:high, ti] = np.maximum(mask[low:high, ti], vals)
            
            # Generate LDL Label
            target_idx = np.arange(65)
            mu = K_real + 1.2 
            ldl_label = np.exp(-0.5 * ((target_idx - mu) / 0.9748) ** 2)
            ldl_label = ldl_label / ldl_label.sum()
            
            yield (spec_resized, torch.tensor(mask).unsqueeze(0).float(), torch.tensor(ldl_label).float(), K_real)

    def __len__(self): 
        return self.epoch_size

class FixedFileDataset(Dataset):
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Validation file '{file_path}' not found! Please run training script to generate it automatically.")
        
        # Load data safely
        self.data = torch.load(file_path, weights_only=True)
        print(f"Loaded {len(self.data)} samples from {file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['spec'], item['mask'], item['ldl'], item['k']

def generate_validation_file(filename="validation_dataset.pt", num_samples=2000):
    print(f"Generating validation dataset ({num_samples} samples)...")
    ds = EndToEndDataset(epoch_size=num_samples, augment=False)
    data_list = []
    for spec, mask, ldl, k in ds:
        data_list.append({'spec': spec, 'mask': mask, 'ldl': ldl, 'k': k})
    torch.save(data_list, filename)
    print(f"Saved validation dataset to {filename}")