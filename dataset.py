import torch
import torchvision
import torchvision.transforms as transforms

class DataSetCifar10():

    def __init__(self,batch_size = 4):
        print('Loading CIFAR-10 DataSet...')
        self.__batch_size = batch_size
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.__trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        self.__trainloader = torch.utils.data.DataLoader(self.__trainset, batch_size=self.__batch_size,
                                                  shuffle=True, num_workers=2)

        self.__testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        self.__testloader = torch.utils.data.DataLoader(self.__testset, batch_size=self.__batch_size,
                                                 shuffle=False, num_workers=2)

        self.__classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        print('Done')

    def set_batch_size(self,batch_size = 4):

        self.__batch_size = batch_size
        self.__trainloader = torch.utils.data.DataLoader(self.__trainset, batch_size=self.batch_size,
                                                       shuffle=True, num_workers=2)

        self.__testloader = torch.utils.data.DataLoader(self.__testset, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=2)

    def get_classes(self):
        return self.__classes

    def get_train(self):
        return self.__trainloader

    def get_test(self):
        return self.__testloader