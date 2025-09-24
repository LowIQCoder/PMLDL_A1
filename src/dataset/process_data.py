import os
import struct
import numpy as np

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

def preprocess_data(base_dir):
    """Preprocess the MNIST data"""
    data_dir = base_dir

    train_images_path = os.path.join(data_dir, "train-images.idx3-ubyte")
    train_labels_path = os.path.join(data_dir, "train-labels.idx1-ubyte")
    test_images_path  = os.path.join(data_dir, "t10k-images.idx3-ubyte")
    test_labels_path  = os.path.join(data_dir, "t10k-labels.idx1-ubyte")

    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)
    test_images  = load_mnist_images(test_images_path)
    test_labels  = load_mnist_labels(test_labels_path)

    df_train = pd.DataFrame(train_images.reshape(-1, 28 * 28))
    df_train['label'] = train_labels

    df_test = pd.DataFrame(test_images.reshape(-1, 28 * 28))
    df_test['label'] = test_labels

    df_train.to_csv("dataset/processed/train.csv", index=False)
    df_test.to_csv("dataset/processed/test.csv", index=False)

    return df_train, df_test


if __name__ == "__main__":
    df_train, df_test = preprocess_data("dataset/raw/MNIST")
    print(df_train.head())
    print(df_test.head())
