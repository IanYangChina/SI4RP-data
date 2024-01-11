from doma.optimiser.adam import Adam
import numpy as np
from doma.engine.configs.macros import DTYPE_NP, EPS

parm = np.asarray(np.random.uniform(1000, 2000), dtype=DTYPE_NP).reshape((1,))  # Young's modulus
optim_E = Adam(parameters_shape=parm.shape,
               cfg={'lr': 10, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})

print(parm)
for _ in range(100):
    grad = np.random.uniform(2.713e-02-4.420e-02, 2.713e-02+4.420e-02, size=parm.shape)
    parm = optim_E.step(parm.copy(), grad.copy())
    parm = np.clip(parm, 1000, 2000)
    print(optim_E.cur_lr, grad, parm)