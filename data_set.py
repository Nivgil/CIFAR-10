import numpy as np
from math import sqrt
import torch
from torch.autograd import Variable
from bokeh.plotting import figure, output_file, show
import torchvision
import torchvision.transforms as transforms
from PIL import ImageFile


class DataSetCifar10(object):

    def __init__(self, batch_size=16, permute=True):
        print('Loading CIFAR-10 DataSet...')
        self._batch_size = batch_size
        self._permute = permute
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, 4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

        self.__trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                       download=True, transform=transform_train)
        self.__trainloader = torch.utils.data.DataLoader(self.__trainset, batch_size=self._batch_size,
                                                         shuffle=permute, num_workers=4)

        self.__testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                      download=True, transform=transform_test)
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


class DataSetCifar100(object):

    def __init__(self, batch_size=16, permute=True):
        print('Loading CIFAR-100 DataSet...')
        self._batch_size = batch_size
        self._permute = permute
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, 4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

        self.__trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                        download=True, transform=transform_train)
        self.__trainloader = torch.utils.data.DataLoader(self.__trainset, batch_size=self._batch_size,
                                                         shuffle=permute, num_workers=4)

        self.__testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                       download=True, transform=transform_test)
        self.__testloader = torch.utils.data.DataLoader(self.__testset, batch_size=self._batch_size,
                                                        shuffle=False, num_workers=4)
        print('Done')

    def set_batch_size(self, batch_size=16):
        self._batch_size = batch_size
        self.__trainloader = torch.utils.data.DataLoader(self.__trainset, batch_size=self._batch_size,
                                                         shuffle=self._permute, num_workers=4)

        self.__testloader = torch.utils.data.DataLoader(self.__testset, batch_size=self._batch_size,
                                                        shuffle=self._permute, num_workers=4)

    def get_train(self):
        return self.__trainloader

    def get_test(self):
        return self.__testloader


class DataSetImageNet(object):

    def __init__(self, batch_size=16, permute=True):
        print('Loading ImageNet DataSet...')
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self._batch_size = batch_size
        self._permute = permute

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

        root = '/home/ehoffer/Datasets/imagenet/'

        self.__trainset = torchvision.datasets.ImageFolder(root=root + 'train', transform=transform_train)
        self.__trainloader = torch.utils.data.DataLoader(self.__trainset, batch_size=self._batch_size,
                                                         shuffle=permute, num_workers=4)

        self.__testset = torchvision.datasets.ImageFolder(root=root + 'val', transform=transform_test)
        self.__testloader = torch.utils.data.DataLoader(self.__testset, batch_size=self._batch_size,
                                                        shuffle=False, num_workers=4)

        print('Done')

    def set_batch_size(self, batch_size=16):
        self._batch_size = batch_size
        self.__trainloader = torch.utils.data.DataLoader(self.__trainset, batch_size=self._batch_size,
                                                         shuffle=self._permute, num_workers=4)

        self.__testloader = torch.utils.data.DataLoader(self.__testset, batch_size=self._batch_size,
                                                        shuffle=self._permute, num_workers=4)

    def get_train(self):
        return self.__trainloader

    def get_test(self):
        return self.__testloader
