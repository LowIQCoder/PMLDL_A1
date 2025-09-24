import os
import struct
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
import pandas as pd

def load_mnist_images(filename):
    """Magic function for loading MNIST images"""
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, 1, rows, cols).astype(np.float32) / 255.0
    return images

def load_mnist_labels(filename):
    """Magic function for loading MNIST labels"""
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def get_mnist_loaders(base_dir, batch_size=64, val_split=0.2):
    """Get DataLoaders for MNIST dataset"""
    df_train = pd.read_csv("dataset/processed/train.csv")
    df_test = pd.read_csv("dataset/processed/test.csv")

    train_dataset = TensorDataset(
        torch.tensor(df_train.drop(columns=["label"]).values, dtype=torch.float32),
        torch.tensor(df_train["label"].values, dtype=torch.long)
    )

    test_dataset = TensorDataset(
        torch.tensor(df_test.drop(columns=["label"]).values, dtype=torch.float32),
        torch.tensor(df_test["label"].values, dtype=torch.long)
    )

    val_size = int(val_split * len(test_dataset))
    test_size = len(test_dataset) - val_size
    val_dataset, test_dataset = random_split(
        test_dataset, [val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
