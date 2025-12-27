# =============================================================================
# Copyright (c) 2025 Yaolin Zou, Xiongfei Tao, Kenan Xiao, Jinkang Shi.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# =============================================================================

import numpy as np

class SignalSimulator:
    def __init__(self, fs=12800, T=0.4):
        self.fs = fs
        self.N = int(fs * T)
        self.t = np.arange(self.N) / fs
        self.T = T
        
    def _generate_noise(self, noise_type='pink'):
        freqs = np.fft.rfftfreq(self.N)
        if noise_type == 'white': S = np.ones_like(freqs)
        elif noise_type == 'pink': S = 1 / (freqs + 1e-4)**0.5
        elif noise_type == 'brown': S = 1 / (freqs + 1e-4)
        elif noise_type == 'blue': S = (freqs + 1e-4)**0.5
        else: S = np.ones_like(freqs)
        phases = np.random.rand(len(freqs)) * 2 * np.pi
        S = S * np.exp(1j * phases)
        noise = np.fft.irfft(S, n=self.N)
        return noise / np.max(np.abs(noise))

    def generate(self):
        f1 = 50 + np.random.uniform(-0.5, 0.5)
        K = int(np.clip(np.random.poisson(lam=15), 3, 50))
        x = np.zeros(self.N)
        freq_tracks = []
        
        for _ in range(K):
            k = np.random.randint(2, 41)
            f0 = k * f1 + np.random.uniform(-2, 2)
            if np.random.rand() > 0.5: 
                f_track = f0 + np.random.uniform(-2, 2) * (self.t / self.T)
            else: 
                f_track = f0 + 0.2 * np.sin(2 * np.pi * 1.5 * self.t)
            
            x += np.random.uniform(0.05, 0.3) * np.sin(2 * np.pi * np.cumsum(f_track) / self.fs)
            freq_tracks.append(f_track)
            
        noise_type = np.random.choice(['white', 'pink', 'brown', 'blue'])
        noise = self._generate_noise(noise_type)
        snr = np.random.uniform(15, 50)
        x += noise / np.linalg.norm(noise) * np.linalg.norm(x) / (10**(snr/20))
        
        if np.random.rand() < 0.4:
            num_bursts = np.random.randint(1, 4)
            for _ in range(num_bursts):
                idx = np.random.randint(0, self.N - 10)
                width = np.random.randint(2, 10)
                x[idx:idx+width] += np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1])
                
        return x.astype(np.float32), K, freq_tracks

class TimeDomainAugmentor:
    def __init__(self, p_gain=0.5, p_noise=0.3, p_cutout=0.4):
        self.p_gain = p_gain
        self.p_noise = p_noise
        self.p_cutout = p_cutout

    def __call__(self, sig):
        if np.random.rand() < self.p_gain: 
            sig = sig * np.random.uniform(0.5, 1.5)
        if np.random.rand() < self.p_noise:
            noise_amp = np.random.uniform(0.01, 0.03) * np.max(np.abs(sig))
            sig = sig + np.random.normal(0, noise_amp, size=sig.shape)
        if np.random.rand() < self.p_cutout:
            N = len(sig)
            width = np.random.randint(int(N * 0.02), int(N * 0.05))
            start = np.random.randint(0, N - width)
            sig[start : start + width] = 0.0
        return sig