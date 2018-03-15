import torch
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_model_parameters(model, dtype):
    parameters = {}
    for name, weight in model.named_parameters():
        parameters[name] = deepcopy(weight.data.type(torch.FloatTensor))
    return parameters


def get_model_gradients(model, dtype):
    gradients = {}
    for name, weight in model.named_parameters():
        gradients[name] = deepcopy(weight.grad.data.type(torch.FloatTensor))
    return gradients


def set_model_paramertes(parameters, model):
    for name, weight in model.named_parameters():
        if torch.cuda.is_available() is True:
            weight.data = parameters[name].cuda()
        else:
            weight.data = parameters[name]
