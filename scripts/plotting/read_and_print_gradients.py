import os
import json
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import matplotlib.pyplot as plt

script_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_path, '..')
import argparse


def main(data_ids=None, contact_level=1, plot=False):
    data_path = os.path.join(script_path, '..', 'gradient-analysis', f'level{contact_level}-12mix')
    fig_path = os.path.join(script_path, '..', 'gradient-analysis', 'figures')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    params = ['E', 'nu', 'ys', 'rho', 'mani_fric', 'g_fric']
    print(f'Param {params}')

    if data_ids is None:
        data_ids = range(19)

    x = []
    y = []
    y_ = []
    y__ = []
    y___ = []
    legends = []
    for n in data_ids:
        if not os.path.exists(os.path.join(data_path, f'grads-mean-{n}.npy')):
            continue

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
        print(f'Mean: {mean}\nLog-magnitude: {np.log(np.abs(mean))}')
        print(f'Std: {std}\nLog-magnitude: {np.log(np.abs(std))}')

        x.append(n)
        legends.append(legend)

        y.append(np.log(np.abs(mean)))
        y_.append(std/mean)
        y__.append(mean)
        y___.append(std)

    if plot:
        print(f"Plots will be saved in {fig_path}")
        y = np.asarray(y)
        y_ = np.asarray(y_)
        y__ = np.asarray(y__)
        y___ = np.asarray(y___)
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
            plt.savefig(os.path.join(fig_path, params[k]+'-order-of-magnitude.pdf'), bbox_inches='tight', dpi=500)
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

            plt.figure()
            for n in range(len(x)):
                if n < 9:
                    m = 0
                else:
                    m = 1
                plt.scatter(x[n], y__[n, k], color=colors[m], marker=markers[m])
                plt.errorbar(x[n], y__[n, k], yerr=y___[n, k], color=colors[m])
            plt.title(params[k]+'-mean')
            plt.xticks(x, legends, rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_path, params[k]+'-mean-std.pdf'), bbox_inches='tight', dpi=500)
            plt.close()


if __name__ == '__main__':
    description = ("This script reads the computed gradients stored in the gradient-analysis folder and prints them."
                   "Optionally, it can plot the gradients."
                   "The gradients are computed for the following parameters: E, nu, ys, rho, mani_fric, g_fric"
                   "Pass multiple data_ids to plot multiple gradients within one plot."
                   "The data_ids can be modified within the script.")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--con_lv', dest='contact_level', type=int, default=0, choices=[1, 2])
    parser.add_argument('--plot', dest='plot', type=str, default=None)
    args = vars(parser.parse_args())
    data_ids = [0]
    main(data_ids=data_ids, contact_level=args['contact_level'], plot=args['plot'])
