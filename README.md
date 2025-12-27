# Adaptive VMD Mode Number Estimation via SA-CRNN

**Official implementation of the paper:** "Adaptive VMD Mode Number Estimation via Spectrum-Attention Enhanced Network for Power Harmonic Analysis", published in *International Journal of Electrical Power & Energy Systems*.

## 📌 Introduction

This repository contains the source code for **SA-CRNN**, an end-to-end deep learning framework designed to adaptively estimate the optimal mode number ($K$) for Variational Mode Decomposition (VMD). The model integrates a **Time-Frequency Attention Module (TFAM)** for feature decoupling and uses a **SafeZone Loss** to ensure robust estimation under high noise.

## 📂 Project Structure

* `simulation.py`: Physics-driven harmonic signal simulator and augmentation.
* `preprocess.py`: Dataset loader (dynamic generation) and STFT preprocessing.
* `tfam.py`: The Attention Module based on ResUNet.
* `model.py`: The main SA-CRNN architecture definition.
* `loss.py`: Custom loss functions (DiceBCELoss, SafeZoneLoss).
* `train.py`: Training script with validation and plotting.
* `end2end_best.pth`: Pre-trained model weights (to be uploaded).

## 🚀 Getting Started

### Prerequisites

* Python >= 3.8
* PyTorch >= 1.10

### Installation

```bash
git clone https://github.com/jiqjijiajdi/Adaptive-VMD-SA-CRNN.git
cd Adaptive-VMD-SA-CRNN
pip install -r requirements.txt
```



### Usage

**1. Training from scratch:** Run the training script. It will automatically generate a fixed validation dataset (`validation_dataset.pt`) on the first run.

Bash

```
python train.py
```

**2. Using the pre-trained model:** If you have downloaded `end2end_best.pth`, ensure it is in the root directory. You can load it in your own inference script using:

Python

```
import torch
from model import EndToEndNet

model = EndToEndNet()
model.load_state_dict(torch.load("end2end_best.pth"))
model.eval()
```

## 📜 Citation

If you find this code useful, please cite our paper:

代码段

```
@article{zou2025adaptive,
  title={Adaptive VMD Mode Number Estimation via Spectrum-Attention Enhanced Network for Power Harmonic Analysis},
  author={Zou, Yaolin and Tao, Xiongfei and Xiao, Kenan and Shi, Jinkang},
  journal={International Journal of Electrical Power & Energy Systems},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License.
Copyright (c) 2025 Yaolin Zou, Xiongfei Tao, Kenan Xiao, Jinkang Shi.



