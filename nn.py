import torch
from torch.autograd import Variable
import data_set
from parameter_server import ParameterServer
from statistics import Statistics
import net_model
from resnet import resnet
from time import time


def train_nn(params):
    workers_number = params.workers_number
    epochs = params.epochs
    gpu = params.gpu

    server, loss_fn, stats_train, stats_test, train_set, test_set, model, dtype = initialization(params)

    print('Training Neural Network...\n' + str(params))
    for t in range(1, epochs + 1):
        tic = time()
        for idx, (data, labels) in enumerate(train_set):
            weights = server.pull(idx % workers_number)
            net_model.set_model_paramertes(weights, model, gpu)
            if gpu is True:
                labels = labels.cuda()
                data = data.cuda()
            data, labels = Variable(data.type(dtype)), Variable(labels)
            y_pred = model(data)
            loss = loss_fn(y_pred, labels)
            model.zero_grad()
            loss.backward()
            server.push(idx % workers_number, net_model.get_model_gradients(model, dtype), t)
        toc = time()
        evaluate_epoch(model, server, stats_train, loss_fn, train_set, t, gpu, dtype, toc - tic, True)
        evaluate_epoch(model, server, stats_test, loss_fn, test_set, t, gpu, dtype, toc - tic, False)
    print('Done')
    return stats_train, stats_test


def initialization(params):
    print('-----------------------')
    print('Initializing...', end='')

    batch_size = params.batch_size
    learning_rate = params.learning_rate
    momentum = params.momentum
    rho = params.rho
    tau = params.tau
    initialization = params.initialization
    workers_number = params.workers_number
    optimizer = params.optimizer
    permute = params.permute
    gpu = params.gpu
    gpu_num = params.gpu_number

    if gpu is True:
        print('Utilizing GPU')
        torch.cuda.set_device(gpu_num)
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    dataset = data_set.DataSetCifar10(batch_size, permute)
    train_set = dataset.get_train()
    test_set = dataset.get_test()

    model = resnet()
    if gpu is True:
        model.cuda()
        # model = torch.nn.DataParallel(model)  # Run on multiple GPUs

    parameters = net_model.get_model_parameters(model, dtype)
    gradients = net_model.get_model_parameters(model, dtype)
    loss_fn = torch.nn.CrossEntropyLoss()
    server = ParameterServer.get_server(optimizer,
                                        learning_rate=learning_rate,
                                        momentum=momentum,
                                        parameters=parameters,
                                        gradients=gradients,
                                        workers_number=workers_number,
                                        rho=rho,
                                        tau=tau)
    stats_train = Statistics.get_statistics('image_classification', params)
    stats_test = Statistics.get_statistics('image_classification', params)
    print('Done')

    return server, loss_fn, stats_train, stats_test, train_set, test_set, model, dtype


def evaluate_epoch(model, server, statistics, loss_fn, data, epoch_num, gpu, dtype, time_t, log=True):
    if epoch_num % 1 == 0 and log == True:
        print('Epoch [{0:1d}], Time [{1:.2f}sec]'.format(epoch_num, time_t), end='')
    total_loss, error = 0, 0
    master_weights = server.get_weights()
    net_model.set_model_paramertes(master_weights, model, gpu)
    for idx, (data, labels) in enumerate(data, 1):
        if gpu is True:
            labels = labels.cuda()
            data = data.cuda()
        data, labels = Variable(data.type(dtype)), Variable(labels)
        y_pred = model(data)
        total_loss = total_loss + loss_fn(y_pred, labels).data[0]
        _, class_pred = torch.max(y_pred, 1)
        error = error + 1 - torch.sum(class_pred.data == labels.data) / len(labels)
    statistics.save_norm(master_weights)
    statistics.save_loss(total_loss / idx)
    statistics.save_error(error / idx)
    if epoch_num % 1 == 0 and log == True:
        print(' , Loss [{0:.5f}] , Error[{1:.2f}%]'.format(total_loss / idx, 100 * error / idx))


# not in use
def evaluate_iteration(model, server, statistics, loss_fn, data, epoch_num, iter_num, gpu, dtype, log=True):
    if epoch_num % 1 == 0 and log is True:
        print('Epoch [{0:1d}], Iteration [{1:1d}]'.format(epoch_num, iter_num), end='')
    total_loss, error = 0, 0
    master_weights = server.get_weights()
    net_model.set_model_paramertes(master_weights, model, gpu)
    for idx, (data, labels) in enumerate(data, 1):
        if gpu is True:
            labels = labels.cuda()
            data = data.cuda()
        data, labels = Variable(data.type(dtype)), Variable(labels)
        y_pred = model(data)
        total_loss = total_loss + loss_fn(y_pred, labels).data[0]
        _, class_pred = torch.max(y_pred, 1)
        error = error + 1 - torch.sum(class_pred.data == labels.data) / len(labels)
    statistics.save_norm(master_weights)
    statistics.save_loss(total_loss / idx)
    statistics.save_error(error / idx)
    if epoch_num % 1 == 0 and log is True:
        print(' , Loss [{0:.5f}] , Error[{1:.2f}%]'.format(total_loss / idx, 100 * error / idx))
