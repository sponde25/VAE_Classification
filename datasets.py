import numpy as np
import torch
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

DEFAULT_DATA_FOLDER = './data'

class Dataset():
    def __init__(self, data_set, data_folder = DEFAULT_DATA_FOLDER):
        super(type(self), self).__init__()

        if data_set == 'mnist':
            self.train_set = dset.MNIST(root=data_folder,
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)

            self.test_set = dset.MNIST(root=data_folder,
                                       train=False,
                                       transform=transforms.ToTensor())

        if data_set == 'cifar10':
            self.composed_transforms = transforms.Compose([[transforms.ToTensor(), transforms.Resize((28, 28)),
                                                     transforms.Normalize([0., 0., 0.], [0.5, 0.5, 0.5])]])
            self.train_set = dset.CIFAR10(root=data_folder,
                                          train=True,
                                          transform=self.composed_transforms,
                                          download=True)

            self.test_set = dset.CIFAR10(root=data_folder,
                                         train=False,
                                         transform=self.composed_transforms)

    def get_train_size(self):
        return len(self.train_set)

    def get_test_size(self):
        return len(self.test_set)

    def get_train_loader(self, batch_size, shuffle=True):
        train_loader = DataLoader(dataset= self.train_set, batch_size=batch_size, shuffle=shuffle, num_workers=8)
        return train_loader

    def get_test_loader(self, batch_size, shuffle=False):
        test_loader = DataLoader(dataset= self.test_set, batch_size=batch_size, shuffle=shuffle, num_workers=8)
        return test_loader


