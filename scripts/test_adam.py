from doma.optimiser.adam import Adam
import numpy as np
from doma.engine.configs.macros import DTYPE_NP, EPS

bounds = (1000, 2000)
mean = 7.381e-02
std = 1.124e-01
parm = np.asarray(np.random.uniform(bounds[0], bounds[1]), dtype=DTYPE_NP).reshape((1,))  # Young's modulus
optim_E = Adam(parameters_shape=parm.shape,
               cfg={'lr': 10, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})

print(parm)
for _ in range(100):
    grad = np.random.uniform(mean-std, mean+std, size=parm.shape)
    parm = optim_E.step(parm.copy(), grad.copy())
    parm = np.clip(parm, bounds[0], bounds[1])
    print(optim_E.cur_lr, grad, parm)