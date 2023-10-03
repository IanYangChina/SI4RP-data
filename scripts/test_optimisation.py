import taichi as ti
import numpy as np
from doma.envs import *
from time import time

mu = np.array([416.6])
lamda = np.array([277.78])

horizon = 120
env = ClayEnv(ptcl_density=1e8, horizon=horizon)
env.reset()
mpm_env = env.mpm_env
init_state = mpm_env.get_state()
given_trajectory = np.zeros(shape=(horizon, mpm_env.agent.action_dim))
# given_trajectory[1:60, -1] = -0.001
# given_trajectory[51:, 0] = 0.001

# forward
t1 = time()
mpm_env.set_state(init_state['state'], grad_enabled=True)
cur_horizon = mpm_env.loss.temporal_range[1]
mpm_env.apply_agent_action_p(np.array((1., 1., 1.)))
for i in range(cur_horizon):
    action = given_trajectory[i]
    mpm_env.step(action)
    env.render(mode='human')

loss_info = mpm_env.get_final_loss()
print(loss_info)
t2 = time()

# # backward
# mpm_env.reset_grad()
# mpm_env.get_final_loss_grad()
# for i in range(cur_horizon-1, -1, -1):
#     action = given_trajectory[i]
#     mpm_env.step_grad(action=action)
# mpm_env.apply_agent_action_p_grad(np.array((1., 1., 1.)))
#
# param_grad = mpm_env.simulator.get_param_grad()
#
# t3 = time()
# print(f'=======> forward: {t2 - t1:.2f}s backward: {t3 - t2:.2f}s')
# print(param_grad)
