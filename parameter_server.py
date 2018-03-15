import torch
import numpy as np
from copy import deepcopy
from math import sqrt


class ParameterServer(object):

    @staticmethod
    def get_server(mode, *args, **kwargs):
        return {'synchronous': SSgd, 'asynchronous': ASgd, 'async_elastic': AESgd}[mode](*args,
                                                                                         **kwargs)

    def __init__(self, learning_rate=None, momentum=None, parameters=None, workers_number=None, gradients=None,
                 effective_batch_size=None, gradient_clipping=None,lr_batch_adjustment=None, **kwargs):
        # TODO: Check Input
        self._norm_type = 2
        self._max_norm = gradient_clipping
        self._momentum = momentum
        self._workers_num_ = workers_number
        self._counter = self._workers_num_

        self._batch_size = effective_batch_size

        self._batch_baseline = 128
        if lr_batch_adjustment is True:
            self._lr_factor = sqrt(self._batch_size / self._batch_baseline)
            self._learning_rate = [self._lr_factor * x for x in learning_rate[0]]
        else:
            self._learning_rate = learning_rate[0]
        self._lr_steps = learning_rate[1]
        self._iteration_counter = 0


        self._gradients = deepcopy(gradients)
        self._velocity = deepcopy(gradients)
        self._weights = deepcopy(parameters)
        # shards - each entry save model parameters entity
        self._shards_weights = [parameters] * self._workers_num_
        self._shards_gradients = [parameters] * self._workers_num_
        self._status = np.zeros((self._workers_num_, 1))
        self.reset_grad()
        self.reset_velocity()

    def reset_grad(self, worker_id=None):
        # TODO: Check Input
        for name, value in self._gradients.items():
            self._gradients[name] = torch.zeros(value.size())
        if worker_id is None:
            for worker_id in range(self._workers_num_):
                for name, value in self._shards_gradients[worker_id].items():
                    self._shards_gradients[worker_id][name] = torch.zeros(value.size())
        else:
            for name, value in self._shards_gradients[worker_id].items():
                self._shards_gradients[worker_id][name] = torch.zeros(value.size())

    def reset_velocity(self):
        # TODO: Check Input
        for name, value in self._velocity.items():
            self._velocity[name] = torch.zeros(value.size())

    def reset_status(self, worker_id=None):
        # TODO: Check Input
        if worker_id is None:
            self._status[:] = np.zeros((self._workers_num_, 1))
        else:
            self._status[worker_id] = np.zeros(1)

    def reset_counter(self):
        self._counter = self._workers_num_

    def _gradient_clipping(self, max_norm, norm_type=2):
        if max_norm == 0:
            return
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        if norm_type == float('inf'):
            total_norm = max(self._gradients[name].abs().max() for name in self._gradients)
        else:
            total_norm = 0
            for name in self._gradients:
                param_norm = self._gradients[name].norm(norm_type)
                total_norm += param_norm ** norm_type
            total_norm = total_norm ** (1. / norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for name in self._gradients:
                self._gradients[name].mul_(clip_coef)

    def update_gradient(self, gradient=None):
        if gradient is not None:
            self._gradients = gradient
        else:
            for name, value in self._gradients.items():
                self._gradients[name] = torch.zeros(value.size())

            for worker_id in range(self._workers_num_):
                for name, value in self._shards_gradients[worker_id].items():
                    self._gradients[name] = self._gradients[name] + value

    def get_weights(self):
        return self._weights

    def get_gradients(self):
        return self._gradients

    def get_learning_rate(self, epoch):
        closest_smaller_epoch = max([x for x in self._lr_steps if x <= self._iteration_counter])
        return self._learning_rate[self._lr_steps.index(closest_smaller_epoch)]

    def update_weights(self):
        raise NotImplementedError

    def push(self, worker_id, gradient, epoch):
        raise NotImplementedError

    def pull(self, worker_id):
        raise NotImplementedError


class SSgd(ParameterServer):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def push(self, worker_id, gradient, epoch):
        # TODO: Check Input
        self._shards_gradients[worker_id] = gradient
        updated = False
        if self._status[worker_id] == 0:
            self._counter -= 1
            self._status[worker_id] = 1

        if self._counter == 0:
            self.update_gradient()
            self._gradient_clipping(self._max_norm,self._norm_type)
            self.update_weights(epoch)
            # self.reset_grad()
            self.reset_status()
            self.reset_counter()
            updated = True
        return updated

    def update_gradient(self, gradient=None):
        for name, value in self._gradients.items():
            self._gradients[name] = torch.zeros(value.size())

        for worker_id in range(self._workers_num_):
            for name, value in self._shards_gradients[worker_id].items():
                self._gradients[name] = self._gradients[name] + value

        for name, value in self._gradients.items():
            self._gradients[name] = self._gradients[name] / self._workers_num_

    def update_weights(self, epoch=None):
        self._iteration_counter += 1
        lr = self.get_learning_rate(epoch)
        for name in self._weights:
            self._velocity[name] = self._momentum * self._velocity[name] - lr * self._gradients[name]
            self._weights[name] = self._weights[name] + self._velocity[name]

        for worker_id in range(self._workers_num_):
            for name, value in self._weights.items():
                self._shards_weights[worker_id][name] = value

    def pull(self, worker_id=None):
        # TODO: Check Input
        return self._shards_weights[worker_id]


class ASgd(ParameterServer):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def push(self, worker_id, gradient, epoch):
        # TODO: Check Input
        # self.reset_grad()
        self._shards_gradients[worker_id] = gradient

        self.update_gradient(gradient)
        self._gradient_clipping(self._max_norm, self._norm_type)
        self.update_weights(worker_id, epoch)
        return True

    def update_weights(self, worker_id=None, epoch=None):
        self._iteration_counter += 1
        lr = self.get_learning_rate(epoch)
        for name in self._weights:
            self._velocity[name] = self._momentum * self._velocity[name] - lr * self._gradients[name]
            self._weights[name] = self._weights[name] + self._velocity[name]

        for name, value in self._weights.items():
            self._shards_weights[worker_id][name] = value

    def pull(self, worker_id=None):
        # TODO: Check Input
        return self._shards_weights[worker_id]


class AESgd(ParameterServer):
    def __init__(self, *args, rho=None, tau=None, **kwargs):
        super().__init__(**kwargs)
        self._rho = rho
        self._tau = tau
        self._shards_velocity = deepcopy(self._shards_gradients)

    def push(self, worker_id, gradient, epoch):
        # TODO: Check Input
        self._shards_gradients[worker_id] = gradient

        if self._status[worker_id] < self._tau:
            self._status[worker_id] = self._status[worker_id] + 1

        self.update_weights(worker_id, epoch)

        if self._status[worker_id] == self._tau:
            self.reset_grad(worker_id)
            self.reset_status(worker_id)
        return True

    def update_velocities(self, worker_id, learning_rate):
        for name in self._weights:
            self._shards_velocity[worker_id][name] = self._momentum * self._shards_velocity[worker_id][name] - \
                                                     learning_rate * self._shards_gradients[worker_id][name]

    def update_weights(self, worker_id=None, epoch=None):
        self._iteration_counter += 1
        lr = self.get_learning_rate(epoch)
        old_worker_weights = deepcopy(self._shards_weights[worker_id])
        if self._status[worker_id] == self._tau:
            for name in self._weights:
                self._shards_weights[worker_id][name] = self._shards_weights[worker_id][name] \
                                                        - lr * self._rho * \
                                                        (old_worker_weights[name] - self._weights[name])
                self._weights[name] = self._weights[name] + lr * self._rho * \
                                      (old_worker_weights[name] - self._weights[name])
        self.update_velocities(worker_id, lr)
        for name in self._weights:
            self._shards_weights[worker_id][name] = self._shards_weights[worker_id][name] + \
                                                    self._shards_velocity[worker_id][name]

    def pull(self, worker_id):
        # TODO: Check Input
        return self._shards_weights[worker_id]
