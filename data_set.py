import numpy as np
from math import sqrt
import torch
from torch.autograd import Variable
from bokeh.plotting import figure, output_file, show
import torchvision
import torchvision.transforms as transforms

# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
dtype = torch.FloatTensor


class DataSetCifar10(object):

    def __init__(self, batch_size=16, permute=True):
        print('Loading CIFAR-10 DataSet...')
        self._batch_size = batch_size
        self._permute = permute
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.__trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                       download=True, transform=transform)
        self.__trainloader = torch.utils.data.DataLoader(self.__trainset, batch_size=self._batch_size,
                                                         shuffle=permute, num_workers=4)

        self.__testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                      download=True, transform=transform)
        self.__testloader = torch.utils.data.DataLoader(self.__testset, batch_size=self._batch_size,
                                                        shuffle=False, num_workers=4)

        self.__classes = ('plane', 'car', 'bird', 'cat',
                          'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        print('Done')

    def set_batch_size(self, batch_size=16):
        self._batch_size = batch_size
        self.__trainloader = torch.utils.data.DataLoader(self.__trainset, batch_size=self._batch_size,
                                                         shuffle=self._permute, num_workers=4)

        self.__testloader = torch.utils.data.DataLoader(self.__testset, batch_size=self._batch_size,
                                                        shuffle=self._permute, num_workers=4)

    def get_classes(self):
        return self.__classes

    def get_train(self):
        return self.__trainloader

    def get_test(self):
        return self.__testloader
