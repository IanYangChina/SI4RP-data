import os
import json
import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt

script_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(script_path, '..', 'gradients-E-nu-ys-rho')
fig_path = os.path.join(data_path, 'figures')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

losses = ['pd_rs', 'pd_sr', 'prd_rs', 'prd_sr', 'hm', 'emd']
params = ['E', 'nu', 'ys', 'rho', 'mani_fric', 'g_fric']
print(f'Param {params}')
n = 0
plt.figure()
while True:
    if not os.path.exists(os.path.join(data_path, f'grads-mean-{n}.npy')):
        break

    loss_cfg = json.load(open(os.path.join(data_path, f'loss-config-{n}.json'), 'r'))
    mean = np.load(os.path.join(data_path, f'grads-mean-{n}.npy'))
    std = np.load(os.path.join(data_path, f'grads-std-{n}.npy'))

    legend = ''
    if loss_cfg['averaging_loss']:
        legend += 'avg-'
    if loss_cfg['exponential_distance']:
        legend += 'exp-'
    if loss_cfg['point_distance_rs_loss']:
        legend += 'pdrs'
    if loss_cfg['point_distance_sr_loss']:
        legend += 'pdsr'
    if loss_cfg['particle_distance_rs_loss']:
        legend += 'prdrs'
    if loss_cfg['particle_distance_sr_loss']:
        legend += 'prdsr'
    if loss_cfg['height_map_loss']:
        legend += 'hm'
    if loss_cfg['emd_point_distance_loss']:
        legend += 'emd_p'
    if loss_cfg['emd_particle_distance_loss']:
        legend += 'emd_pr'

    print(f'Loss {n}: {legend}')
    print(f'Mean: {mean}')
    print(f'Std: {std}')

    n += 1
