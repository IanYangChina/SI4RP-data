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

x = []
y = []
y_ = []
e = []
legends = []
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
    print(f'Mean: {mean}, OoM: {np.log(np.abs(mean))}')
    print(f'Std: {std}, OoM: {np.log(np.abs(std))}')
    
    x.append(n)
    legends.append(legend)
    n += 1
    
    y.append(np.log(np.abs(mean)))
    y_.append(std/mean)
    
y = np.asarray(y)
y_ = np.asarray(y_)
colors = ['b', 'g']
markers = ['+', 'x']
for k in range(6):
    plt.figure()
    for n in range(len(x)):
        if n < 9:
            m = 0
        else:
            m = 1
        plt.scatter(x[n], y[n, k], color=colors[m], marker=markers[m])
    plt.title(params[k]+'-log-abs-mean')
    plt.xticks(x, legends, rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, params[k]+'-mean.pdf'), bbox_inches='tight', dpi=500)
    plt.close()
    
    plt.figure()
    for n in range(len(x)):
        if n < 9:
            m = 0
        else:
            m = 1
        plt.scatter(x[n], y_[n, k], color=colors[m], marker=markers[m])
    plt.title(params[k]+'-coeff-variation')
    plt.xticks(x, legends, rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, params[k]+'-coeff-variation.pdf'), bbox_inches='tight', dpi=500)
    plt.close()
