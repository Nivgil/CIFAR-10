import torch
import numpy as np
from torch.autograd import Variable
import data_set
import random
from parameter_server import ParameterServer
from statistics import Statistics
import net_model
from resnet import resnet
from alexnet import alexnet
from time import time


def train_nn(params):
    workers_number = params.workers_number
    epochs = params.epochs

    server, loss_fn, stats_train, stats_test, train_set, test_set, model, dtype = initialization(params)
    gpu = torch.cuda.is_available()

    print('Training Neural Network...\n' + str(params))
    for t in range(1, epochs + 1):
        tic = time()
        for idx, (data, labels) in enumerate(train_set):
            weights = server.pull(idx % workers_number)
            net_model.set_model_paramertes(weights, model)
            if gpu is True:
                labels = labels.cuda()
                data = data.cuda()
            data, labels = Variable(data.type(dtype)), Variable(labels)
            y_pred = model(data)
            loss = loss_fn(y_pred, labels)
            model.zero_grad()
            loss.backward()
            server.push(idx % workers_number, net_model.get_model_gradients(model, dtype), t)
        train_toc = time()
        train_loss, train_error = evaluate_epoch(model, server, stats_train, loss_fn, train_set, gpu, dtype)
        test_loss, test_error = evaluate_epoch(model, server, stats_test, loss_fn, test_set, gpu, dtype)
        evaluate_toc = time()
        print('ID {0}: Epoch [{1:1d}], Train Time [{2:.2f}sec], Evaluation Time [{3:.2f}sec], '
              'Train Loss [{4:.5f}], Train Error[{5:.2f}%], Validation Loss [{6:.5f}], Validation Error[{7:.2f}%]'
              .format(params.simulation_id, t, train_toc - tic, evaluate_toc - train_toc, train_loss, train_error,
                      test_loss, test_error))
    print('Simulation {} Done'.format(params.simulation_id))
    return stats_train, stats_test


def initialization(params):
    print('-----------------------')
    print('Initializing...', end='')

    batch_size = params.batch_size
    learning_rate = params.learning_rate
    momentum = params.momentum
    rho = params.rho
    tau = params.tau
    workers_number = params.workers_number
    optimizer = params.optimizer
    permute = params.permute
    gpu_num = params.gpu_number
    gradient_clipping = params.gradient_clipping
    lr_batch_adjustment = params.lr_batch_adjustment

    if torch.cuda.is_available() is True:
        print('Utilizing GPU')
        torch.cuda.set_device(gpu_num)
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    if params.data_set == 'cifar10':
        dataset = data_set.DataSetCifar10(batch_size, permute)
        model = resnet(num_classes=10, depth=56, wide_factor=1)
    if params.data_set == 'image_net':
        dataset = data_set.DataSetImageNet(batch_size, permute)
        model = alexnet()
    if params.data_set == 'cifar100':
        dataset = data_set.DataSetCifar100(batch_size, permute)
        model = resnet(num_classes=100, depth=56, wide_factor=1)

    train_set = dataset.get_train()
    test_set = dataset.get_test()

    if torch.cuda.is_available() is True:
        model.cuda()
        # model = torch.nn.DataParallel(model)  # Run on multiple GPUs

    parameters = net_model.get_model_parameters(model, dtype)
    gradients = net_model.get_model_parameters(model, dtype)
    loss_fn = torch.nn.CrossEntropyLoss()
    if optimizer == 'synchronous':
        effective_batch_size = batch_size * workers_number
    else:
        effective_batch_size = batch_size
    server = ParameterServer.get_server(optimizer,
                                        learning_rate=learning_rate,
                                        momentum=momentum,
                                        parameters=parameters,
                                        gradients=gradients,
                                        workers_number=workers_number,
                                        rho=rho,
                                        tau=tau,
                                        effective_batch_size=effective_batch_size,
                                        gradient_clipping=gradient_clipping,
                                        lr_batch_adjustment=lr_batch_adjustment)
    stats_train = Statistics.get_statistics('image_classification', params)
    stats_test = Statistics.get_statistics('image_classification', params)
    print('Done')

    return server, loss_fn, stats_train, stats_test, train_set, test_set, model, dtype


def evaluate_epoch(model, server, statistics, loss_fn, dataset, gpu, dtype):
    total_loss, error = 0, 0
    master_weights = server.get_weights()
    master_gradients = server.get_gradients()
    net_model.set_model_paramertes(master_weights, model)
    for idx, (data, labels) in enumerate(dataset, 1):
        if gpu is True:
            labels = labels.cuda()
            data = data.cuda()
        data, labels = Variable(data.type(dtype)), Variable(labels)
        y_pred = model(data)
        total_loss = total_loss + loss_fn(y_pred, labels).data[0]
        _, class_pred = torch.max(y_pred, 1)
        error = error + 1 - torch.sum(class_pred.data == labels.data) / len(labels)
    statistics.save_weight_norm(master_weights)
    statistics.save_gradient_norm(master_gradients)
    statistics.save_loss(total_loss / idx)
    statistics.save_error(error / idx)
    loss = total_loss / idx
    error = 100 * error / idx
    return loss, error
