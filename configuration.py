import os
import numpy as np
import json
import logging
from math import ceil

optimizers = {'1': 'synchronous', '2': 'asynchronous', '3': 'elastic', '4': 'async_elastic'}

CONFIGURATIONS_DIR = os.path.join(os.path.dirname(__file__))
with open(os.path.join(CONFIGURATIONS_DIR, 'unicode.json'), 'r') as symbols:
    symbols = json.load(symbols)


class Configuration(object):
    def __init__(self, params_dic=None, simulation_id=None):
        CONFIGURATIONS_DIR = os.path.join(os.path.dirname(__file__))
        if params_dic is None:
            with open(os.path.join(CONFIGURATIONS_DIR, 'parameters.json'), 'r') as param_file:
                params_dic = json.load(param_file)
        np.random.seed()
        self.simulation_name = params_dic['sim_name']
        self.epochs = params_dic['epochs']
        self.batch_size = params_dic['batch_size']  # mini batch size
        self.learning_rate = params_dic['learning_rate']
        self.momentum = params_dic['momentum']
        self.rho = params_dic['rho']
        self.tau = params_dic['tau']
        self.workers_number = params_dic['workers_number']  # Turns Batch_Size <- mini_batch_size * workers_number
        if simulation_id is None:
            self.simulation_id = round(1e4 * np.random.rand())
        else:
            self.simulation_id = simulation_id
        self.simulation_number = params_dic['sim_num']
        self.folder_name = 'simulation_{}'.format(self.simulation_number)
        self.live_results = eval(params_dic['live_results'])
        self.optimizer = params_dic['optimizer']
        self.gpu_number = params_dic['gpu_number']
        self.permute = eval(params_dic['data_permutation'])
        self.gradient_clipping = params_dic['gradient_clipping']
        self.lr_batch_adjustment = eval(params_dic['lr_batch_adjustment'])
        self.data_set = params_dic['data_set']

        self.iterations_per_epoch = ceil(50000 / self.batch_size)
        if self.optimizer == 'synchronous':
            self.iterations_per_epoch = round(self.iterations_per_epoch / self.workers_number)

        path_name = os.path.join(CONFIGURATIONS_DIR, self.folder_name)

        if not os.path.exists(path_name):
            os.mkdir(path_name)
        log_name = os.path.join(path_name, self.simulation_name + str(self.simulation_id) + '.log')
        if os.path.exists(os.path.join(path_name,log_name)):
            raise Exception('Log file already exists')
        logging.basicConfig(format='%(message)s', filename=log_name, level=logging.ERROR)

    def __str__(self):
        string = '----------------------------------\n'
        string = string + ('simulation name   - {}\n'.format(self.simulation_name))
        string = string + ('simulation id     - {}\n'.format(self.simulation_id))
        string = string + ('optimizer         - {}\n'.format(self.optimizer))
        string = string + ('workers_number    - {}\n'.format(self.workers_number))
        string = string + ('batch_size        - {}\n'.format(self.batch_size))
        string = string + ('epochs            - {}\n'.format(self.epochs))
        string = string + ('learning_rate     - {}\n'.format(self.learning_rate))
        string = string + ('lr_factorization  - {}\n'.format(self.lr_batch_adjustment))
        string = string + ('gradient_clipping - {}\n'.format(self.gradient_clipping))
        string = string + ('rho               - {}\n'.format(self.rho))
        string = string + ('tau               - {}\n'.format(self.tau))
        string = string + ('momentum          - {}\n'.format(self.momentum))
        string = string + ('live_results      - {}\n'.format(self.live_results))
        string = string + ('data_permutation  - {}\n'.format(self.permute))
        string = string + ('gpu_number        - {}\n'.format(self.gpu_number))
        return string

    def __repr__(self):
        string = '----------------------------------\n'
        string = string + ('simulation name   - {}\n'.format(self.simulation_name))
        string = string + ('simulation id     - {}\n'.format(self.simulation_id))
        string = string + ('optimizer         - {}\n'.format(self.optimizer))
        string = string + ('workers_number    - {}\n'.format(self.workers_number))
        string = string + ('batch_size        - {}\n'.format(self.batch_size))
        string = string + ('epochs            - {}\n'.format(self.epochs))
        string = string + ('learning_rate     - {}\n'.format(self.learning_rate))
        string = string + ('lr_factorization  - {}\n'.format(self.lr_batch_adjustment))
        string = string + ('gradient_clipping - {}\n'.format(self.gradient_clipping))
        string = string + ('rho               - {}\n'.format(self.rho))
        string = string + ('tau               - {}\n'.format(self.tau))
        string = string + ('momentum          - {}\n'.format(self.momentum))
        string = string + ('live_results      - {}\n'.format(self.live_results))
        string = string + ('data_permutation  - {}\n'.format(self.permute))
        string = string + ('gpu_number        - {}\n'.format(self.gpu_number))
        return string

    def to_dict(self):
        params_dict = dict()
        params_dict['simulation_name'] = self.simulation_name
        params_dict['simulation_id'] = self.simulation_id
        params_dict['data_set'] = self.data_set
        params_dict['optimizer'] = self.optimizer
        params_dict['workers_number'] = self.workers_number
        params_dict['batch_size'] = self.batch_size
        params_dict['epochs'] = self.epochs
        params_dict['learning_rate'] = self.learning_rate
        params_dict['gradient_clipping'] = self.gradient_clipping
        params_dict['lr_batch_adjustment'] = self.lr_batch_adjustment
        params_dict['rho'] = self.rho
        params_dict['tau'] = self.tau
        params_dict['momentum'] = self.momentum
        params_dict['permute'] = self.permute

        return params_dict
