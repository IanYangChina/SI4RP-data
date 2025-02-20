import numpy as np
from doma.engine.configs.macros import DTYPE_NP
from doma.optimiser.adam import Optimizer


class RMSprop(Optimizer):
    def __init__(self, parameters_shape, cfg):
        super(RMSprop, self).__init__(parameters_shape, cfg)
        self.normaliser = np.zeros(self.parameters_shape).astype(DTYPE_NP) + 1e-8
        self.normaliser_ = np.zeros(self.parameters_shape).astype(DTYPE_NP) + 1e-8
        self.iter = 0

    def _step(self, parameters, grads):
        beta = self.cfg['beta']
        self.iter += 1
        self.normaliser_ = self.normaliser * 1.0
        self.normaliser = beta * self.normaliser + (1 - beta) * grads * grads
        cur_lr = self.lr / (np.sqrt(self.normaliser) + 1e-8)
        return parameters - cur_lr * grads

    def reverse_normaliser(self):
        self.normaliser = self.normaliser_


class AbsProp(Optimizer):
    def __init__(self, parameters_shape, cfg):
        super(AbsProp, self).__init__(parameters_shape, cfg)
        self.normaliser = np.zeros_like(self.parameters_shape).astype(DTYPE_NP) + 1e-8
        self.iter = 0

    def _step(self, parameters, grads):
        cur_lr = self.lr / (np.abs(grads) + 1e-8)
        return parameters - cur_lr * grads
