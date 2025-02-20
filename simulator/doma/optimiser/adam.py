import numpy as np
from doma.engine.configs.macros import DTYPE_NP


class Optimizer:
    def __init__(self, parameters_shape, cfg):
        self.cfg = cfg
        self.lr = self.cfg['lr']
        print('====> lr: ', self.lr)
        self.init_lr = self.cfg['lr']
        self.parameters_shape = parameters_shape

    def initialize(self):
        raise NotImplementedError

    def _step(self, parameters, grads):
        raise NotImplementedError

    def step(self, parameters, grads):
        return self._step(parameters, grads)


class GD(Optimizer):
    def __init__(self, parameters_shape, cfg):
        super(GD, self).__init__(parameters_shape, cfg)

    def initialize(self):
        pass

    def _step(self, parameters, grads):
        return parameters - self.lr * grads


class Adam(Optimizer):
    def __init__(self, parameters_shape, cfg):
        super(Adam, self).__init__(parameters_shape, cfg)
        self.momentum_buffer = np.zeros(self.parameters_shape).astype(DTYPE_NP)
        self.v_buffer = np.zeros_like(self.momentum_buffer).astype(DTYPE_NP)
        self.iter = 0
        self.initialize()
        self.cur_lr = np.ones(self.parameters_shape).astype(DTYPE_NP) * self.lr

    def initialize(self):
        self.momentum_buffer *= 0
        self.v_buffer *= 0
        self.iter = 0

    def _step(self, parameters, grads):
        beta_1 = self.cfg['beta_1']
        beta_2 = self.cfg['beta_2']
        epsilon = self.cfg['epsilon']
        m_t = beta_1 * self.momentum_buffer + (1 - beta_1) * grads  # updates the moving averages of the gradient
        assert not np.any(np.isinf(m_t)) or np.any(np.isnan(m_t)), f'm_t is inf or nan with momentum_buffer: {self.momentum_buffer} and grads: {grads}'
        v_t = beta_2 * self.v_buffer + (1 - beta_2) * (grads * grads)  # updates the moving averages of the squared gradient
        assert not np.any(np.isinf(v_t)) or np.any(np.isnan(v_t)), f'v_t is inf or nan with v_buffer: {self.v_buffer} and grads: {grads}'

        self.momentum_buffer[:] = m_t
        self.v_buffer[:] = v_t

        m_cap = m_t / (1 - (beta_1 ** (self.iter + 1)))  # calculates the bias-corrected estimates
        assert not np.any(np.isinf(m_cap)) or np.any(np.isnan(m_cap)), f'm_cap is inf or nan with m_t: {m_t} and iter: {self.iter}'
        v_cap = v_t / (1 - (beta_2 ** (self.iter + 1)))  # calculates the bias-corrected estimates
        assert not np.any(np.isinf(v_cap)) or np.any(np.isnan(v_cap)), f'v_cap is inf or nan with v_t: {v_t} and iter: {self.iter}'

        self.iter += 1
        new_pram = parameters - (self.lr * m_cap) / (np.sqrt(v_cap) + epsilon)
        self.cur_lr = (self.lr * m_cap) / (np.sqrt(v_cap) + epsilon) / grads
        assert not np.any(np.isinf(new_pram)) or np.any(np.isnan(new_pram)), f'new_pram is inf or nan with parameters: {parameters} and m_cap: {m_cap}'
        return new_pram
