import torch
import numpy as np
from copy import deepcopy


class ParameterServer(object):

    @staticmethod
    def get_server(mode, *args, **kwargs):
        return {'synchronous': SSgd, 'asynchronous': ASgd, 'async_elastic': AESgd}[mode](*args,
                                                                                         **kwargs)

    def __init__(self, learning_rate=None, momentum=None, parameters=None, workers_number=None, gradients=None,
                 **kwargs):
        # TODO: Check Input
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._workers_num_ = workers_number
        self._batch_complete = False
        self._counter = self._workers_num_

        self._lr = (1e-1, 1e-2, 1e-3, 1e-4)
        self._ep = (1, 81, 122, 164)

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

    def update_gradient(self):
        for name, value in self._gradients.items():
            self._gradients[name] = torch.zeros(value.size())

        for worker_id in range(self._workers_num_):
            for name, value in self._shards_gradients[worker_id].items():
                self._gradients[name] = self._gradients[name] + value

    def get_weights(self):
        return self._weights

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
            self.update_weights(epoch)
            self.reset_grad()
            self.reset_status()
            self.reset_counter()
            updated = True
        return updated

    def update_gradient(self):
        for name, value in self._gradients.items():
            self._gradients[name] = torch.zeros(value.size())

        for worker_id in range(self._workers_num_):
            for name, value in self._shards_gradients[worker_id].items():
                self._gradients[name] = self._gradients[name] + value

        for name, value in self._gradients.items():
            self._gradients[name] = self._gradients[name] / self._workers_num_

    def update_weights(self, epoch=None):
        if epoch >= self._ep[2]:
            lr = self._lr[2]
        elif epoch >= self._ep[1]:
            lr = self._lr[1]
        else:
            lr = self._lr[0]
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
        self.reset_grad()
        self._shards_gradients[worker_id] = gradient

        self.update_gradient()
        self.update_weights(worker_id, epoch)
        return True

    def update_weights(self, worker_id=None, epoch=None):
        if epoch >= self._ep[2]:
            lr = self._lr[2]
        elif epoch >= self._ep[1]:
            lr = self._lr[1]
        else:
            lr = self._lr[0]
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
        if epoch >= self._ep[2]:
            lr = self._lr[2]
        elif epoch >= self._ep[1]:
            lr = self._lr[1]
        else:
            lr = self._lr[0]
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
