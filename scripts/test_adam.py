from doma.optimiser.adam import Adam
import numpy as np
from doma.engine.configs.macros import DTYPE_NP, EPS
"""
E_range = (1e4, 3e5)
nu_range = (0.01, 0.49)
yield_stress_range = (1e3, 2e4)
rho_range = (1000, 2000)
mf_range = (0.01, 2.0)
gf_range = (0.01, 2.0)
"""
bounds = (1e4, 3e5)
mean = -3.799e-03
std = 6.329e-03
parm = np.asarray(np.random.uniform(bounds[0], bounds[1]), dtype=DTYPE_NP).reshape((1,))  # Young's modulus
optim_E = Adam(parameters_shape=parm.shape,
               cfg={'lr': 5e3, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})

print(parm)
for _ in range(100):
    grad = np.random.uniform(mean-std, mean+std, size=parm.shape)
    parm = optim_E.step(parm.copy(), grad.copy())
    parm = np.clip(parm, bounds[0], bounds[1])
    print(optim_E.cur_lr, grad, parm)

"""
'lr_E': 5e3,
'lr_nu': 1e-2,
'lr_yield_stress': 5e2,
'lr_rho': 10,
'lr_manipulator_friction': 1e-2,
'lr_ground_friction': 1e-2,
"""