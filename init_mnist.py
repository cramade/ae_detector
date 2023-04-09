import os
import gzip
import torch
import urllib.request
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class MNIST():
    def __init__(self, batch_size=100):
        dir_name = "mnist_data"
        self.data_train = datasets.MNIST(root = dir_name, train=True, download=True, transform=transforms.ToTensor())
        self.data_test = datasets.MNIST(root = dir_name, train=False, download=True, transform=transforms.ToTensor())
        self.train_loader = torch.utils.data.DataLoader(dataset = self.data_train, batch_size = batch_size, shuffle = True, drop_last = True)
        self.test_loader = torch.utils.data.DataLoader(dataset = self.data_test, batch_size = 1, shuffle = True)

    def print(self):
        print(f"MNIST Data")
        print(f"Train data size : {len(self.data_train)}")
        print(f"Test data size : {len(self.data_test)}")
        return "MNIST data"


# mnist = MNIST()
# mnist.print()