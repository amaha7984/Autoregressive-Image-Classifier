# Autoregressive-Image-Classifier
Multimodal Autoregressive Image Model for Binary Classification

This repository contains the implementation of an **Autoregressive Image Model (AIM)** for **binary image classification**, built upon the ideas presented in the paper:  
[**Multimodal Autoregressive Pre-training of Large Vision Encoders**](https://arxiv.org/abs/2411.14402), published by [Apple ML Research](https://github.com/apple/ml-aim).

---

## Overview

This project leverages an **autoregressive, multimodal vision model** for binary classification tasks. The core model is designed for scalability and performance on multimodal vision-language tasks, following recent advancements in autoregressive pre-training. The code is implemented using **PyTorch** and supports **multi-GPU distributed training** using Distributed Data Parallel (DDP).

---

## Installation

- Clone the repository:
   ```bash
   git clone https://github.com/amaha7984/Autoregressive-Image-Classifier.git
   cd AutoRegressive-Image-Classifier
   ```
- Create a Python virtual environment (optional)
  ```bash
   python3.9 -m venv myvenv
   source myvenv/bin/activate
  ```
- Install required dependencies and packages
  ```bash
  pip3.9 install -r requirements.txt
  ```
- Train the Autoregressive-Image-Classifier:
```bash
python3.9 train.py --train_path ./datasets/train/ --val_path ./datasets/val/ --total_epochs 150 --batch_size 256
```
The model's weight will be stored at `./saved_models/`.
  




