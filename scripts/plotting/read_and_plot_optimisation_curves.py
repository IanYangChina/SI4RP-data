import os
import json
import numpy as np
from drl_implementation.agent.utils import plot as plot
import matplotlib as mpl
from copy import deepcopy as dcp

mpl.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

np.set_printoptions(2, suppress=True)

import argparse

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams["font.weight"] = "normal"
cwd = os.path.dirname(os.path.realpath(__file__))
cwd = os.path.join(cwd, '..')
loss_types = ['point_distance_sr',
              'point_distance_rs',
              'chamfer_loss_pcd',
              'particle_distance_sr',
              'particle_distance_rs',
              'chamfer_loss_particle',
              'height_map_loss_pcd',
              'emd_point_distance_loss',
              'emd_particle_distance_loss'
              ]
params = ['E', 'nu', 'yield_stress', 'rho', 'mf', 'gf']


def read_final_params(run_ids, contact_level=0, dataset='12mix'):
    for run in run_ids:
        for seed in [0, 1, 2]:
            path = os.path.join(cwd, '..', 'optimisation-results',
                                f'level{contact_level}-{dataset}-run{run}-logs',
                                f'seed-{seed}', 'final_params.npy')
            parameters = np.load(path)
            for i in range(len(params)):
                print(f'Run {run}, Seed {seed}: {params[i]}: {parameters[i]}')


def read_losses(run_ids, contact_level=0, dataset='12mix',
                extra_seeds=False, man_init=False, save_meanstd=False):
    """generate mean and deviation data from tensorboard logs"""
    assert contact_level in [1, 2]
    assert dataset in ['12mix', '6mix', '1rec', '1round', '1cyl', 'slime', 'soil']
    assert len(run_ids) > 0
    dir_prefix = f'level{contact_level}-{dataset}'
    for run_id in run_ids:
        run_dir = os.path.join(cwd, '..', 'optimisation-results',
                               f'{dir_prefix}-run{run_id}-logs', )
        if man_init:
            run_dir = os.path.join(cwd, '..', 'optimisation-results',
                                   f'{dir_prefix}-run{run_id}-man-init-logs', )
        if extra_seeds:
            run_dir = os.path.join(cwd, '..', 'optimisation-results',
                                   f'{dir_prefix}-run{run_id}-extra-seeds-logs', )
        save_data_dir = os.path.join(run_dir, 'data')
        os.makedirs(save_data_dir, exist_ok=True)
        sub_data_dict = {
                'training': {
                    'point_distance_sr': [],
                    'point_distance_rs': [],
                    'chamfer_loss_pcd': [],
                    'particle_distance_sr': [],
                    'particle_distance_rs': [],
                    'chamfer_loss_particle': [],
                    'height_map_loss_pcd': [],
                    'emd_point_distance_loss': [],
                    'emd_particle_distance_loss': []
                },
                'validation': {
                    'point_distance_sr': [],
                    'point_distance_rs': [],
                    'chamfer_loss_pcd': [],
                    'particle_distance_sr': [],
                    'particle_distance_rs': [],
                    'chamfer_loss_particle': [],
                    'height_map_loss_pcd': [],
                    'emd_point_distance_loss': [],
                    'emd_particle_distance_loss': []
                },
                'parameters': {
                    'E': [],
                    'nu': [],
                    'yield_stress': [],
                    'rho': [],
                    'mf': [],
                    'gf': []
                }
            }
        data_dict = {}
        if extra_seeds:
            seeds = [3, 4, 5, 6, 7, 8, 9, 10]
        elif man_init:
            seeds = [0, 1, 2, 3, 4, 5, 6, 7]
        else:
            seeds = [0, 1, 2]
        for seed in seeds:
            data_dict[f'seed-{seed}'] = dcp(sub_data_dict)
        for seed in seeds:
            seed_dir = os.path.join(run_dir, f'seed-{str(seed)}')
            for filename in os.listdir(seed_dir):
                if filename[:5] == 'event':
                    for event in summary_iterator(os.path.join(seed_dir, filename)):
                        for v in event.summary.value:
                            if v.tag[:5] == 'Loss/':
                                if v.tag[5:] == 'point_distance_sr':
                                    data_dict[f'seed-{seed}']['training']['point_distance_sr'].append(v.simple_value)
                                elif v.tag[5:] == 'point_distance_rs':
                                    data_dict[f'seed-{seed}']['training']['point_distance_rs'].append(v.simple_value)
                                elif v.tag[5:] == 'chamfer_loss_pcd':
                                    data_dict[f'seed-{seed}']['training']['chamfer_loss_pcd'].append(v.simple_value)
                                elif v.tag[5:] == 'particle_distance_sr':
                                    data_dict[f'seed-{seed}']['training']['particle_distance_sr'].append(v.simple_value)
                                elif v.tag[5:] == 'particle_distance_rs':
                                    data_dict[f'seed-{seed}']['training']['particle_distance_rs'].append(v.simple_value)
                                elif v.tag[5:] == 'chamfer_loss_particle':
                                    data_dict[f'seed-{seed}']['training']['chamfer_loss_particle'].append(
                                        v.simple_value)
                                elif v.tag[5:] == 'height_map_loss_pcd':
                                    data_dict[f'seed-{seed}']['training']['height_map_loss_pcd'].append(v.simple_value)
                                elif v.tag[5:] == 'emd_point_distance_loss':
                                    data_dict[f'seed-{seed}']['training']['emd_point_distance_loss'].append(
                                        v.simple_value)
                                elif v.tag[5:] == 'emd_particle_distance_loss':
                                    data_dict[f'seed-{seed}']['training']['emd_particle_distance_loss'].append(
                                        v.simple_value)
                                else:
                                    pass
                            elif v.tag[:16] == 'Validation loss/':
                                if v.tag[16:] == 'point_distance_sr':
                                    data_dict[f'seed-{seed}']['validation']['point_distance_sr'].append(v.simple_value)
                                elif v.tag[16:] == 'point_distance_rs':
                                    data_dict[f'seed-{seed}']['validation']['point_distance_rs'].append(v.simple_value)
                                elif v.tag[16:] == 'chamfer_loss_pcd':
                                    data_dict[f'seed-{seed}']['validation']['chamfer_loss_pcd'].append(v.simple_value)
                                elif v.tag[16:] == 'particle_distance_sr':
                                    data_dict[f'seed-{seed}']['validation']['particle_distance_sr'].append(
                                        v.simple_value)
                                elif v.tag[16:] == 'particle_distance_rs':
                                    data_dict[f'seed-{seed}']['validation']['particle_distance_rs'].append(
                                        v.simple_value)
                                elif v.tag[16:] == 'chamfer_loss_particle':
                                    data_dict[f'seed-{seed}']['validation']['chamfer_loss_particle'].append(
                                        v.simple_value)
                                elif v.tag[16:] == 'height_map_loss_pcd':
                                    data_dict[f'seed-{seed}']['validation']['height_map_loss_pcd'].append(
                                        v.simple_value)
                                elif v.tag[16:] == 'emd_point_distance_loss':
                                    data_dict[f'seed-{seed}']['validation']['emd_point_distance_loss'].append(
                                        v.simple_value)
                                elif v.tag[16:] == 'emd_particle_distance_loss':
                                    data_dict[f'seed-{seed}']['validation']['emd_particle_distance_loss'].append(
                                        v.simple_value)
                                else:
                                    pass
                            elif v.tag[:6] == 'Param/':
                                if v.tag[6:] == 'E':
                                    data_dict[f'seed-{seed}']['parameters']['E'].append(v.simple_value)
                                elif v.tag[6:] == 'nu':
                                    data_dict[f'seed-{seed}']['parameters']['nu'].append(v.simple_value)
                                elif v.tag[6:] == 'yield_stress':
                                    data_dict[f'seed-{seed}']['parameters']['yield_stress'].append(v.simple_value)
                                elif v.tag[6:] == 'rho':
                                    data_dict[f'seed-{seed}']['parameters']['rho'].append(v.simple_value)
                                elif v.tag[6:] == 'manipulator_friction':
                                    data_dict[f'seed-{seed}']['parameters']['mf'].append(v.simple_value)
                                elif v.tag[6:] == 'ground_friction':
                                    data_dict[f'seed-{seed}']['parameters']['gf'].append(v.simple_value)
                                else:
                                    pass
                            else:
                                pass

        json.dump(data_dict, open(os.path.join(save_data_dir, 'raw_loss.json'), 'w'))

        if save_meanstd:
            for loss_type in loss_types:
                # training
                data = []
                for seed in seeds:
                    d = np.array(data_dict[f'seed-{seed}']['training'][loss_type])
                    data.append(d)
                plot.get_mean_and_deviation(np.array(data),
                                            save_data=True,
                                            file_name=os.path.join(save_data_dir, f'training-{loss_type}.json'))
                # validation
                validation_data = []
                for seed in seeds:
                    d = data_dict[f'seed-{seed}']['validation'][loss_type]
                    validation_data.append(np.array(d))
                plot.get_mean_and_deviation(np.array(validation_data),
                                            save_data=True,
                                            file_name=os.path.join(save_data_dir, f'validation-{loss_type}.json'))


def plot_legends(extra_seeds=False, heightmap=False):
    if extra_seeds:
        legends = [f'seed-{n}' for n in range(8)]
        plt.rcParams.update({'font.size': 40})
        plot.smoothed_plot_mean_deviation(
            file=os.path.join(cwd, '..', 'figures', 'result-figs', 'legend-extra-seeds.pdf'),
            legend_file=os.path.join(cwd, '..', 'figures', 'result-figs', 'legend-extra-seeds.pdf'),
            horizontal_lines=None, linestyle='--', linewidth=15, handlelength=1,
            legend=legends, legend_ncol=4, legend_frame=False,
            legend_bbox_to_anchor=(1.6, 1.7),
            legend_loc='upper right',
            data_dict_list=[None for _ in range(len(legends))], legend_only=True)
    else:
        plt.rcParams.update({'font.size': 40})
        if heightmap:
            legends = [
                'PCD CD',
                'PRT EMD',
                'Height Map',
                'Negated Height Map'
            ]
            file_name = '-hm'
        else:
            legends = [
                'PCD CD',
                'PRT CD',
                'PCD EMD',
                'PRT EMD',
            ]
            file_name = ''

        plot.smoothed_plot_mean_deviation(
            file=os.path.join(cwd, '..', 'figures', 'result-figs', 'legend.pdf'),
            legend_file=os.path.join(cwd, '..', 'figures', 'result-figs',
                                     f'legend{file_name}.pdf'),
            horizontal_lines=None, linestyle='--', linewidth=15, handlelength=1.1,
            legend=legends, legend_ncol=4, legend_frame=False,
            legend_bbox_to_anchor=(5, 2),
            legend_loc='upper right',
            data_dict_list=[None for _ in range(len(legends))], legend_only=True)


def plot_loss_param_curves(contact_level=1):
    assert contact_level in [1, 2]
    loss_types = ['chamfer_loss_pcd',
                  'chamfer_loss_particle',
                  'emd_point_distance_loss',
                  'emd_particle_distance_loss',
                  'height_map_loss_pcd'
                  ]
    params = ['E', 'nu', 'yield_stress', 'rho', 'mf', 'gf']
    if contact_level == 1:
        params = ['E', 'nu', 'yield_stress', 'rho']
    plt.rcParams.update({'font.size': 40})
    result_dir = os.path.join(cwd, '..', 'figures', 'result-figs')
    os.makedirs(result_dir, exist_ok=True)

    fig, axes = plt.subplots(5, 5, figsize=(5 * 5, 5 * 4))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    datasets = ['12mix', '6mix', '1rec', '1round', '1cyl']
    dataset_name = ['12-mix', '6-mix', '1-rec.', '1-round', '1-cyl.']
    case = 'validation'
    for dataset_id in range(5):
        dataset = datasets[dataset_id]
        dir_prefix = f'level{contact_level}-{dataset}'
        if dataset == '12mix':
            y_axis_off = False
        else:
            y_axis_off = True
        for loss_id in range(5):
            title = None
            loss_type = loss_types[loss_id]
            if loss_type == 'chamfer_loss_pcd':
                title = dataset_name[dataset_id]
                y_label = 'PCD\nCD'
                ylim_valid = [8690, 9650]
                if contact_level == 2:
                    ylim_valid = [16000, 19800]
            elif loss_type == 'chamfer_loss_particle':
                y_label = 'PRT\nCD'
                ylim_valid = [5880, 6480]
                if contact_level == 2:
                    ylim_valid = [10250, 15100]
            elif loss_type == 'emd_point_distance_loss':
                y_label = 'PCD\nEMD'
                ylim_valid = [1170, 1390]
                if contact_level == 2:
                    ylim_valid = [1670, 2800]
            elif loss_type == 'emd_particle_distance_loss':
                y_label = 'PRT\nEMD'
                ylim_valid = [5240, 7200]
                if contact_level == 2:
                    ylim_valid = [10750, 26000]
            elif loss_type == 'height_map_loss_pcd':
                y_label = 'Height\nMap'
                ylim_valid = [1970, 2400]
                if contact_level == 2:
                    ylim_valid = [2500, 4000]
            else:
                raise ValueError('Unknown loss type')

            yticks = (round(ylim_valid[0] * 1.03),
                      round(ylim_valid[1] * 0.97))

            color_pool = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'k']
            datas = []
            colors = []
            linestyles = []
            linewidths = []
            alphas = []
            n = 0
            for run_id in [0, 1, 2, 3]:
                run_dir = os.path.join(cwd, '..', 'optimisation-results', f'{dir_prefix}-run{run_id}-logs', 'data')
                losses = json.load(open(os.path.join(run_dir, 'raw_loss.json'), 'rb'))
                for seed in [0, 1, 2]:
                    datas.append(losses[f'seed-{seed}'][case][loss_type])
                    colors.append(color_pool[n])
                    linestyles.append(':')
                    linewidths.append(4)
                    alphas.append(1)
                with open(os.path.join(run_dir, f'{case}-{loss_type}.json'), 'rb') as f:
                    d = json.load(f)
                datas.append(d['mean'])
                colors.append(color_pool[n])
                linestyles.append('-')
                linewidths.append(4)
                alphas.append(1)
                n += 1

            max_y = np.max(np.max(datas))
            min_y = np.min(np.min(datas))
            if case == 'training':
                ylim_valid = (min_y * 0.995, max_y * 1.005)
                delta_y = (max_y - min_y) / 30
                yticks = (round(min_y + delta_y),
                          round((min_y + max_y) / 2),
                          round(max_y - delta_y))

            # Plotting
            axes[loss_id, dataset_id].set_xlabel(None)
            axes[loss_id, dataset_id].set_xticks([])
            if y_axis_off:
                axes[loss_id, dataset_id].set_ylabel(None)
                axes[loss_id, dataset_id].set_yticks([])
            else:
                axes[loss_id, dataset_id].set_ylabel(y_label, rotation='horizontal', horizontalalignment='left')
                axes[loss_id, dataset_id].yaxis.set_label_coords(-0.8, 0.3)
                if yticks is not None:
                    axes[loss_id, dataset_id].set_yticks(yticks)
            if ylim_valid[0] is not None:
                axes[loss_id, dataset_id].set_ylim(ylim_valid)
            if title is not None:
                axes[loss_id, dataset_id].set_title(title)

            window = 10
            for t in range(len(datas)):
                N = len(datas[t])
                x = [i for i in range(N)]
                if window != 0:
                    running_avg = np.empty(N)
                    for n in range(N):
                        running_avg[n] = np.mean(datas[t][max(0, n - window):(n + 1)])
                else:
                    running_avg = datas[t]
                axes[loss_id, dataset_id].plot(x, running_avg,
                                               c=colors[t], linestyle=linestyles[t],
                                               linewidth=linewidths[t], alpha=alphas[t])
    plt.savefig(os.path.join(result_dir, f'loss-curves-level{contact_level}.pdf'),
                bbox_inches='tight', pad_inches=0, dpi=500)
    plt.close(fig)

    fig, axes = plt.subplots(len(params), 5, figsize=(5 * 5, len(params) * 4))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    for dataset_id in range(5):
        dataset = datasets[dataset_id]
        dir_prefix = f'level{contact_level}-{dataset}'
        if dataset == '12mix':
            y_axis_off = False
        else:
            y_axis_off = True

        for p_id in range(len(params)):
            p = params[p_id]
            if p == 'E':
                ylim_valid = [1e4 - 15000, 3e5 + 15000]
                yticks = (1e4, 3e5)
                yticklabels = ['1e4', '3e5']
                y_label = 'Young\'s\nModulus\n$E$ (kPa)'
            elif p == 'nu':
                ylim_valid = [-0.03, 0.52]
                yticks = (0, 0.5)
                yticklabels = ['0', '0.5']
                y_label = 'Poisson\'s\nRatio\n$\\nu$'
            elif p == 'yield_stress':
                ylim_valid = [1e3 - 1000, 2e4 + 1000]
                yticks = (1e3, 2e4)
                yticklabels = ['1e3', '2e4']
                y_label = 'Yield\nStress\n$\\sigma_y$ (Pa)'
            elif p == 'rho':
                ylim_valid = [950, 2050]
                yticks = (1000, 2000)
                yticklabels = ['1e3', '2e3']
                y_label = 'Material\nDensity\n$\\rho$ (kg/m$^3$)'
            elif p == 'mf':
                ylim_valid = [-0.15, 2.15]
                yticks = (0, 2)
                yticklabels = ['0.0', '2.0']
                y_label = 'Manipulator\nFriction\n$\\mu_m$'
            elif p == 'gf':
                ylim_valid = [-0.15, 2.15]
                yticks = (0, 2)
                yticklabels = ['0.0', '2.0']
                y_label = 'Table\nFriction\n$\\mu_t$'
            else:
                raise ValueError('Unknown parameter')

            color_pool = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
            datas = []
            colors = []
            linewidths = []
            n = 0
            for run_id in [0, 1, 2, 3]:
                run_dir = os.path.join(cwd, '..', 'optimisation-results', f'{dir_prefix}-run{run_id}-logs', 'data')
                losses = json.load(open(os.path.join(run_dir, 'raw_loss.json'), 'rb'))
                for seed in [0, 1, 2]:
                    datas.append(losses[f'seed-{seed}']['parameters'][p])
                    colors.append(color_pool[n])
                    linewidths.append(2)
                n += 1

            # Plotting
            axes[p_id, dataset_id].set_xlabel(None)
            axes[p_id, dataset_id].set_xticks([])
            if y_axis_off:
                axes[p_id, dataset_id].set_ylabel(None)
                axes[p_id, dataset_id].set_yticks([])
            else:
                axes[p_id, dataset_id].set_ylabel(y_label, rotation='horizontal', horizontalalignment='left')
                axes[p_id, dataset_id].yaxis.set_label_coords(-0.8, 0.25)
                if yticks is not None:
                    axes[p_id, dataset_id].set_yticks(yticks, yticklabels)
            if ylim_valid[0] is not None:
                axes[p_id, dataset_id].set_ylim(ylim_valid)

            window = 10
            for t in range(len(datas)):
                N = len(datas[t])
                x = [i for i in range(N)]
                if window != 0:
                    running_avg = np.empty(N)
                    for n in range(N):
                        running_avg[n] = np.mean(datas[t][max(0, n - window):(n + 1)])
                else:
                    running_avg = datas[t]
                axes[p_id, dataset_id].plot(x, running_avg,
                                               c=colors[t], linestyle='-',
                                               linewidth=linewidths[t], alpha=1)

    plt.savefig(os.path.join(result_dir, f'param-curves-level{contact_level}.pdf'),
                bbox_inches='tight', pad_inches=0, dpi=500)
    plt.close(fig)


def plot_loss_param_curves_extra(man_init=False, heightmap=False):
    loss_types = ['chamfer_loss_pcd',
                  'chamfer_loss_particle',
                  'emd_point_distance_loss',
                  'emd_particle_distance_loss',
                  'height_map_loss_pcd'
                  ]
    params = ['E', 'nu', 'yield_stress', 'rho']
    plt.rcParams.update({'font.size': 40})
    result_dir = os.path.join(cwd, '..', 'figures', 'result-figs')
    os.makedirs(result_dir, exist_ok=True)

    fig, axes = plt.subplots(5, 3, figsize=(5 * 3, 5 * 4))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    datasets = ['1rec', '1round', '1cyl']
    dataset_name = ['1-rec.', '1-round', '1-cyl.']
    case = 'validation'
    for dataset_id in range(len(datasets)):
        dataset = datasets[dataset_id]
        dir_prefix = f'level1-{dataset}'
        if dataset == '1rec':
            y_axis_off = False
        else:
            y_axis_off = True
        for loss_id in range(5):
            title = None
            loss_type = loss_types[loss_id]
            if loss_type == 'chamfer_loss_pcd':
                title = dataset_name[dataset_id]
                y_label = 'PCD\nCD'
                ylim_valid = [8720, 10280]
            elif loss_type == 'chamfer_loss_particle':
                y_label = 'PRT\nCD'
                ylim_valid = [5850, 6600]
            elif loss_type == 'emd_point_distance_loss':
                y_label = 'PCD\nEMD'
                ylim_valid = [1160, 1490]
            elif loss_type == 'emd_particle_distance_loss':
                y_label = 'PRT\nEMD'
                ylim_valid = [5240, 7200]
            elif loss_type == 'height_map_loss_pcd':
                y_label = 'Height\nMap'
                ylim_valid = [1970, 2480]
            else:
                raise ValueError('Unknown loss type')

            yticks = (round(ylim_valid[0] * 1.03),
                      round(ylim_valid[1] * 0.97))

            color_pool = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'k']
            datas = []
            colors = []
            linestyles = []
            linewidths = []
            alphas = []
            if not heightmap:
                for run_id in [3]:
                    run_dir = os.path.join(cwd, '..', 'optimisation-results',
                                           f'{dir_prefix}-run{run_id}-extra-seeds-logs', 'data')
                    losses = json.load(open(os.path.join(run_dir, 'raw_loss.json'), 'rb'))
                    n = 0
                    for seed in [3, 4, 5, 6, 7, 8, 9, 10]:
                        datas.append(losses[f'seed-{seed}'][case][loss_type])
                        colors.append(color_pool[n])
                        linestyles.append('-')
                        linewidths.append(2)
                        alphas.append(1)
                        n += 1

                    if man_init:
                        run_dir = os.path.join(cwd, '..', 'optimisation-results',
                                               f'{dir_prefix}-run{run_id}-man-init-logs', 'data')
                        losses = json.load(open(os.path.join(run_dir, 'raw_loss.json'), 'rb'))
                        n = 0
                        for seed in [0, 1, 2, 3, 4, 5, 6, 7]:
                            datas.append(losses[f'seed-{seed}'][case][loss_type])
                            colors.append(color_pool[n])
                            linestyles.append(':')
                            linewidths.append(3)
                            alphas.append(1)
                            n += 1
            else:
                n = 0
                for run_id in [0, 3, 4, 5]:
                    run_dir = os.path.join(cwd, '..', 'optimisation-results',
                                           f'{dir_prefix}-run{run_id}-logs', 'data')
                    losses = json.load(open(os.path.join(run_dir, 'raw_loss.json'), 'rb'))
                    for seed in [0, 1, 2]:
                        datas.append(losses[f'seed-{seed}'][case][loss_type])
                        colors.append(color_pool[n])
                        linestyles.append(':')
                        linewidths.append(2)
                        alphas.append(1)
                    with open(os.path.join(run_dir, f'{case}-{loss_type}.json'), 'rb') as f:
                        d = json.load(f)
                    datas.append(d['mean'])
                    colors.append(color_pool[n])
                    linestyles.append('-')
                    linewidths.append(4)
                    alphas.append(1)
                    n += 1

            max_y = np.max(np.max(datas))
            min_y = np.min(np.min(datas))
            if case == 'training':
                ylim_valid = (min_y * 0.995, max_y * 1.005)
                delta_y = (max_y - min_y) / 30
                yticks = (round(min_y + delta_y),
                          round((min_y + max_y) / 2),
                          round(max_y - delta_y))

            # Plotting
            axes[loss_id, dataset_id].set_xlabel(None)
            axes[loss_id, dataset_id].set_xticks([])
            if y_axis_off:
                axes[loss_id, dataset_id].set_ylabel(None)
                axes[loss_id, dataset_id].set_yticks([])
            else:
                axes[loss_id, dataset_id].set_ylabel(y_label, rotation='horizontal', horizontalalignment='left')
                axes[loss_id, dataset_id].yaxis.set_label_coords(-0.8, 0.3)
                if yticks is not None:
                    axes[loss_id, dataset_id].set_yticks(yticks)
            if ylim_valid[0] is not None:
                axes[loss_id, dataset_id].set_ylim(ylim_valid)
            if title is not None:
                axes[loss_id, dataset_id].set_title(title)

            window = 10
            for t in range(len(datas)):
                N = len(datas[t])
                x = [i for i in range(N)]
                if window != 0:
                    running_avg = np.empty(N)
                    for n in range(N):
                        running_avg[n] = np.mean(datas[t][max(0, n - window):(n + 1)])
                else:
                    running_avg = datas[t]
                axes[loss_id, dataset_id].plot(x, running_avg,
                                               c=colors[t], linestyle=linestyles[t],
                                               linewidth=linewidths[t], alpha=alphas[t])
    if heightmap:
        plt.savefig(os.path.join(result_dir, f'{case}-loss-curves-level1-heightmap.pdf'),
                    bbox_inches='tight', pad_inches=0, dpi=500)
    else:
        plt.savefig(os.path.join(result_dir, f'{case}-loss-curves-level1-extra-seeds.pdf'),
                    bbox_inches='tight', pad_inches=0, dpi=500)
    plt.close(fig)

    fig, axes = plt.subplots(len(params), 3, figsize=(5 * 3, len(params) * 4))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    for dataset_id in range(len(datasets)):
        dataset = datasets[dataset_id]
        dir_prefix = f'level1-{dataset}'
        if dataset == '1rec':
            y_axis_off = False
        else:
            y_axis_off = True

        for p_id in range(len(params)):
            p = params[p_id]
            if p == 'E':
                ylim_valid = [1e4 - 15000, 3e5 + 15000]
                yticks = (1e4, 3e5)
                yticklabels = ['1e4', '3e5']
                y_label = 'Young\'s\nModulus\n$E$ (kPa)'
            elif p == 'nu':
                ylim_valid = [-0.03, 0.52]
                yticks = (0, 0.5)
                yticklabels = ['0', '0.5']
                y_label = 'Poisson\'s\nRatio\n$\\nu$'
            elif p == 'yield_stress':
                ylim_valid = [1e3 - 1000, 2e4 + 1000]
                yticks = (1e3, 2e4)
                yticklabels = ['1e3', '2e4']
                y_label = 'Yield\nStress\n$\\sigma_y$ (Pa)'
            elif p == 'rho':
                ylim_valid = [950, 2050]
                yticks = (1000, 2000)
                yticklabels = ['1e3', '2e3']
                y_label = 'Material\nDensity\n$\\rho$ (kg/m$^3$)'
            else:
                raise ValueError('Unknown parameter')

            color_pool = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'k']
            datas = []
            colors = []
            linewidths = []
            linestyles = []
            if not heightmap:
                for run_id in [3]:
                    run_dir = os.path.join(cwd, '..', 'optimisation-results', f'{dir_prefix}-run{run_id}-extra-seeds-logs', 'data')
                    losses = json.load(open(os.path.join(run_dir, 'raw_loss.json'), 'rb'))
                    n = 0
                    for seed in [3, 4, 5, 6, 7, 8, 9, 10]:
                        datas.append(losses[f'seed-{seed}']['parameters'][p])
                        colors.append(color_pool[n])
                        linewidths.append(2)
                        linestyles.append('-')
                        n += 1
                    if man_init:
                        run_dir = os.path.join(cwd, '..', 'optimisation-results',
                                               f'{dir_prefix}-run{run_id}-man-init-logs', 'data')
                        losses = json.load(open(os.path.join(run_dir, 'raw_loss.json'), 'rb'))
                        n = 0
                        for seed in [0, 1, 2, 3, 4, 5, 6, 7]:
                            datas.append(losses[f'seed-{seed}']['parameters'][p])
                            colors.append(color_pool[n])
                            linewidths.append(3)
                            linestyles.append(':')
                            n += 1
            else:
                n = 0
                for run_id in [0, 3, 4, 5]:
                    run_dir = os.path.join(cwd, '..', 'optimisation-results', f'{dir_prefix}-run{run_id}-logs', 'data')
                    losses = json.load(open(os.path.join(run_dir, 'raw_loss.json'), 'rb'))
                    for seed in [0, 1, 2]:
                        datas.append(losses[f'seed-{seed}']['parameters'][p])
                        colors.append(color_pool[n])
                        linewidths.append(2)
                        linestyles.append('-')
                    n += 1

            # Plotting
            axes[p_id, dataset_id].set_xlabel(None)
            axes[p_id, dataset_id].set_xticks([])
            if y_axis_off:
                axes[p_id, dataset_id].set_ylabel(None)
                axes[p_id, dataset_id].set_yticks([])
            else:
                axes[p_id, dataset_id].set_ylabel(y_label, rotation='horizontal', horizontalalignment='left')
                axes[p_id, dataset_id].yaxis.set_label_coords(-0.8, 0.25)
                if yticks is not None:
                    axes[p_id, dataset_id].set_yticks(yticks, yticklabels)
            if ylim_valid[0] is not None:
                axes[p_id, dataset_id].set_ylim(ylim_valid)

            window = 10
            for t in range(len(datas)):
                N = len(datas[t])
                x = [i for i in range(N)]
                if window != 0:
                    running_avg = np.empty(N)
                    for n in range(N):
                        running_avg[n] = np.mean(datas[t][max(0, n - window):(n + 1)])
                else:
                    running_avg = datas[t]
                axes[p_id, dataset_id].plot(x, running_avg,
                                               c=colors[t], linestyle=linestyles[t],
                                               linewidth=linewidths[t], alpha=1)
    if heightmap:
        plt.savefig(os.path.join(result_dir, f'param-curves-level1-heightmap.pdf'),
                    bbox_inches='tight', pad_inches=0, dpi=500)
    else:
        plt.savefig(os.path.join(result_dir, f'param-curves-level1-extra-seeds.pdf'),
                    bbox_inches='tight', pad_inches=0, dpi=500)
    plt.close(fig)


def collect_best_validation_losses(run_ids, contact_level=0, dataset='12mix'):
    case = 'validation'
    assert contact_level in [1, 2]
    assert dataset in ['12mix', '6mix', '1rec', '1round', '1cyl']
    assert len(run_ids) > 0
    params = ['E', 'nu', 'yield_stress', 'rho', 'gf', 'mf']
    if contact_level == 1:
        params = ['E', 'nu', 'yield_stress', 'rho']
    dir_prefix = f'level{contact_level}-{dataset}'
    data_dict = dict()
    for run_id in run_ids:
        run_dir = os.path.join(cwd, '..', 'optimisation-results', f'{dir_prefix}-run{run_id}-logs', 'data')
        losses = json.load(open(os.path.join(run_dir, 'raw_loss.json'), 'rb'))
        mean_hm = 5000000
        p_values = []
        for seed in [0, 1, 2]:
            mean_hm_ = np.mean(losses[f'seed-{seed}'][case]['height_map_loss_pcd'][-10:])
            if mean_hm_ < mean_hm:
                mean_hm = mean_hm_
                p_values = []
                for p in params:
                    p_values.append(losses[f'seed-{seed}']['parameters'][p][-1])
        data_dict.update({
            f'run{run_id}': {
                'min_mean_hm': mean_hm,
                'best_p_values': p_values
            }
        })

    with open(os.path.join(cwd, '..', 'optimisation-results', 'figures', dir_prefix,
                           'best-seed-visualisation', 'validation_losses.json'), 'w') as f:
        json.dump(data_dict, f)


def collect_beset_long_horizon_motion_losses(run_ids, save=False, print_loss=False):
    dicts = []
    for dataset in ['12mix', '6mix', '1rec', '1round', '1cyl']:
        dir_prefix = f'level2-{dataset}'

        data_dict = {
            'rectangle-motion': {},
            'round-motion': {},
            'cylinder-motion': {}
        }
        for run_id in run_ids:
            run_dir = os.path.join(cwd, '..', 'optimisation-results', 'figures', dir_prefix,
                                   f'run{run_id}')
            for seed in [0, 1, 2]:
                seed_dir = os.path.join(run_dir, f'seed{seed}')
                if os.path.isdir(seed_dir):
                    break
            for motion_agent in ['rectangle', 'round', 'cylinder']:
                data_dir_1 = os.path.join(seed_dir, f'long_motion-{motion_agent}')
                losses_1 = json.load(open(os.path.join(data_dir_1, 'loss_info.json'), 'rb'))
                hm_loss_1 = losses_1['height_map_loss_pcd']
                data_dict[f'{motion_agent}-motion'].update({
                        f'data-0-run{run_id}': hm_loss_1
                })

        dicts.append(data_dict)
        if save:
            with open(os.path.join(cwd, '..', 'optimisation-results', 'figures', dir_prefix,
                                   'best-seed-visualisation', 'long_motion_losses.json'), 'w') as f:
                json.dump(data_dict, f)

    if print_loss:
        for motion_agent in ['rectangle', 'round', 'cylinder']:
            for data_id in ['data-0']:
                dataset_avg = np.zeros(shape=(5,))
                for run_id in run_ids:
                    print(f'{motion_agent} motion {data_id} run{run_id}')
                    print(f'{dicts[0][f"{motion_agent}-motion"][f"{data_id}-run{run_id}"]:.2f} &', end=' ')
                    print(f'{dicts[1][f"{motion_agent}-motion"][f"{data_id}-run{run_id}"]:.2f} &', end=' ')
                    print(f'{dicts[2][f"{motion_agent}-motion"][f"{data_id}-run{run_id}"]:.2f} &', end=' ')
                    print(f'{dicts[3][f"{motion_agent}-motion"][f"{data_id}-run{run_id}"]:.2f} &', end=' ')
                    print(f'{dicts[4][f"{motion_agent}-motion"][f"{data_id}-run{run_id}"]:.2f} &', end=' ')
                    avg = (dicts[0][f"{motion_agent}-motion"][f"{data_id}-run{run_id}"] +
                           dicts[1][f"{motion_agent}-motion"][f"{data_id}-run{run_id}"] +
                           dicts[2][f"{motion_agent}-motion"][f"{data_id}-run{run_id}"] +
                           dicts[3][f"{motion_agent}-motion"][f"{data_id}-run{run_id}"] +
                           dicts[4][f"{motion_agent}-motion"][f"{data_id}-run{run_id}"]) / 5
                    print(f'{avg:.2f} \n')
                    dataset_avg[0] += dicts[0][f"{motion_agent}-motion"][f"{data_id}-run{run_id}"]
                    dataset_avg[1] += dicts[1][f"{motion_agent}-motion"][f"{data_id}-run{run_id}"]
                    dataset_avg[2] += dicts[2][f"{motion_agent}-motion"][f"{data_id}-run{run_id}"]
                    dataset_avg[3] += dicts[3][f"{motion_agent}-motion"][f"{data_id}-run{run_id}"]
                    dataset_avg[4] += dicts[4][f"{motion_agent}-motion"][f"{data_id}-run{run_id}"]

                print(f'{motion_agent} motion {data_id} avg')
                print(f'{dataset_avg[0] / 4:.2f} &', end=' ')
                print(f'{dataset_avg[1] / 4:.2f} &', end=' ')
                print(f'{dataset_avg[2] / 4:.2f} &', end=' ')
                print(f'{dataset_avg[3] / 4:.2f} &', end=' ')
                print(f'{dataset_avg[4] / 4:.2f} \n')


if __name__ == '__main__':
    description = ("This scripts contains various functions that were used to read the optimisation logs and plot the figures embedded in the paper."
                   "If you want to run them yourselves, modify this script by uncommenting the function calls and providing the necessary arguments."
                   "To run some of these functions you need to install the DRL_implementation package as instructed in the README.md file.")
    parser = argparse.ArgumentParser(description=description)
    """
    The read_losses function reads the tensorboard logs and store the training and validation statistics as well as their means and standard deviations.
    Examples:
    """
    # read_final_params(run_ids=[0], contact_level=2, dataset='soil')
    # read_losses(run_ids=[3], contact_level=1, dataset='1cyl', extra_seeds=False, man_init=True, save_meanstd=True)
    # read_losses(run_ids=[3], contact_level=1, dataset='1rec', extra_seeds=False, man_init=True, save_meanstd=True)
    # read_losses(run_ids=[3], contact_level=1, dataset='1round', extra_seeds=False, man_init=True, save_meanstd=True)
    """
    The plot_loss_param_curves() function plots the curves of the training and validation losses and the parameters using the statistics saved by the read_losses() function.
    Examples:
    """
    # plot_loss_param_curves(contact_level=1)
    # plot_loss_param_curves(contact_level=2)
    # plot_loss_param_curves_extra(heightmap=True)
    # plot_legends(heightmap=True)
    """
    The collect_best_validation_losses() and collect_beset_long_horizon_motion_losses() functions collect the best losses of simulating the validation motions and long horizon motions.
    The best validation losses are determined by the mean of the last 10 validation heightmap losses during training.
    The best long horizon motion losses are determined by the heightmap loss of the last frame of the long horizon motion.
    Their results were summarised as two tables in the paper.
    Examples:
    """
    # collect_best_validation_losses(run_ids=[0, 1, 2, 3], contact_level=1, dataset='12mix')
    # collect_best_validation_losses(run_ids=[0, 1, 2, 3], contact_level=2, dataset='12mix')
    collect_beset_long_horizon_motion_losses(run_ids=[0, 1, 2, 3], print_loss=True)
