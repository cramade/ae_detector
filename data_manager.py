import torch
import urllib.request
import torchvision
from easydict import EasyDict
from torchvision import datasets, transforms

class MNIST():
    def __init__(self, batch_size=100):
        dir_name = "mnist_data"
        self.data_train = datasets.MNIST(root = dir_name, train=True, download=True, transform=transforms.ToTensor())
        self.data_test = datasets.MNIST(root = dir_name, train=False, download=True, transform=transforms.ToTensor())
        self.train_loader = torch.utils.data.DataLoader(dataset = self.data_train, batch_size = batch_size, shuffle = True, drop_last = True)
        self.test_loader = torch.utils.data.DataLoader(dataset = self.data_test, batch_size = 1, shuffle = False)

    def get_data(self):
        return EasyDict(train=self.train_loader, test=self.test_loader)

    def print(self):
        print(f"MNIST Data")
        print(f"Train data size : {len(self.data_train)}")
        print(f"Test data size : {len(self.data_test)}")
        return "MNIST data"

class AdvImages(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.x_data = images
        self.y_data = labels
    
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = self.y_data[idx]
        return x,y

#mnist = MNIST()
#mnist.print()