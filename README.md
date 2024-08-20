# ResNet-50 CIFAR-10 Training with PyTorch

This project implements a deep learning pipeline to train a ResNet-50 model on the CIFAR-10 dataset using PyTorch. It includes features like early stopping, detailed TensorBoard logging for both per-batch and per-epoch metrics, and model evaluation on a test set.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [TensorBoard Visualization](#tensorboard-visualization)
- [Project Structure](#project-structure)
- [License](#license)

## Installation

To set up the project environment, we use `rye`, a Python package manager. Follow the steps below to get started:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/resnet-cifar10-pytorch.git
    cd resnet-cifar10-pytorch
    ```

2. **Install the Required Packages**:
    Install the necessary Python packages using `rye`:
    ```bash
    rye add torch torchvision tensorboard numpy
    ```

## Usage

### Training the Model

Run the training script to start training the ResNet-50 model on the CIFAR-10 dataset:
```bash
python train.py
