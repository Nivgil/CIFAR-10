import logging
from multiprocessing import Process
import pickle
import nn
from configuration import Configuration, symbols
import json
import os


def main():
    CONFIGURATIONS_DIR = os.path.join(os.path.dirname(__file__))
    with open(os.path.join(CONFIGURATIONS_DIR, 'parameters.json'), 'r') as param_file:
        params_dic = json.load(param_file)

    workers = (2, 4, 6, 8, 10)
    gpu_number = (0, 1)
    for idx, worker_num in enumerate(workers):
        params_dic['workers_num'] = worker_num
        params_dic['gpu_number'] = gpu_number[idx % len(gpu_number)]
        p = Process(target=exec_unit, args=params_dic)
        p.start()


def exec_unit(params_dic=None):
    params = Configuration(params_dic)

    stats_train, stats_test = nn.train_nn(params)
    with open(params.folder_name + '/' + params.simulation_name + str(params.simulation_id), 'wb') as pickle_out:
        pickle.dump((stats_test, stats_train), pickle_out)
    logging.info(json.dumps(params.to_dict()))


if __name__ == '__main__':
    exec_unit()
