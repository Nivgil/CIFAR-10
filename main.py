import logging
from multiprocessing import Process
import pickle
import nn
from configuration import Configuration, symbols
import json
import os
import numpy as np
import torch
import random

def main():
    CONFIGURATIONS_DIR = os.path.join(os.path.dirname(__file__))
    with open(os.path.join(CONFIGURATIONS_DIR, 'parameters.json'), 'r') as param_file:
        params_dic = json.load(param_file)

    workers = (1, 1, 1)
    gpu_number = (1, 2, 3)
    for idx, worker_num in enumerate(workers):
        params_dic['workers_number'] = worker_num
        params_dic['gpu_number'] = gpu_number[idx % len(gpu_number)]
        p = Process(target=exec_unit, args=(params_dic, idx,))
        p.start()
    p.join()
    # exec_unit(params_dic, idx)


def exec_unit(params_dic=None, simulation_id=None):
    torch.manual_seed(214)
    torch.cuda.manual_seed(214)
    random.seed(214)
    np.random.seed(214)
    params = Configuration(params_dic, simulation_id)
    stats_train, stats_test = nn.train_nn(params)
    with open(params.folder_name + '/' + params.simulation_name + str(params.simulation_id), 'wb') as pickle_out:
        pickle.dump((stats_test, stats_train), pickle_out)
    logging.info(json.dumps(params.to_dict()))


if __name__ == '__main__':
    main()
