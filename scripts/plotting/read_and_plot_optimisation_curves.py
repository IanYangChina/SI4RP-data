import os
import json
import numpy as np
from drl_implementation.agent.utils import plot as plot
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

np.set_printoptions(2, suppress=True)

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams["font.weight"] = "normal"
cwd = os.path.dirname(os.path.realpath(__file__))
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


def read_plot_params_scatter(run_ids, param_set=0, dist_type='Euclidean', fewshot=True, oneshot=False):
    """Read/plot params"""
    assert len(run_ids) > 0
    legends = [
        f'{dist_type}: PCD chamfer - 0',
        f'{dist_type}: PCD chamfer - 1',
        f'{dist_type}: PCD chamfer - 2',
        f'{dist_type}: Particle chamfer - 0',
        f'{dist_type}: Particle chamfer - 1',
        f'{dist_type}: Particle chamfer - 2',
        f'{dist_type}: PCD emd - 0',
        f'{dist_type}: PCD emd - 1',
        f'{dist_type}: PCD emd - 2',
        f'{dist_type}: Particle emd - 0',
        f'{dist_type}: Particle emd - 1',
        f'{dist_type}: Particle emd - 2',
        f'{dist_type}: Height map - 0',
        f'{dist_type}: Height map - 1',
        f'{dist_type}: Height map - 2'
    ]
    print(f'[Warning]: make sure the run ids are in the correct order based on \n'
          f'{legends[0]}\n{legends[3]}\n{legends[6]}\n{legends[9]}\n{legends[12]}')
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'k']
    param_names = ['E', 'nu', 'yield stress', 'rho', 'mf', 'gf']
    if fewshot:
        assert not oneshot
        dir_prefix = f'optimisation-fewshot-param{param_set}'
    elif oneshot:
        dir_prefix = f'optimisation-oneshot-param{param_set}'
    else:
        dir_prefix = f'optimisation-param{param_set}'
    for p_id in range(4):
        n = 0
        if p_id != 3:
            plt.figure(figsize=(8, 2.5))
        else:
            plt.figure(figsize=(8, 5))
        for k in range(len(run_ids)):
            run_id = run_ids[k]
            run_dir = os.path.join(cwd, '..', f'{dir_prefix}-run{run_id}-logs', )
            values = []
            for seed in [0, 1, 2]:
                seed_dir = os.path.join(run_dir, f'seed-{str(seed)}')
                params = np.load(os.path.join(seed_dir, 'final_params.npy'))
                plt.scatter(n, params.flatten()[p_id], color=colors[k])
                n += 1
        plt.title(f'Final value of {param_names[p_id]}')
        plt.xticks(np.arange(15), legends, rotation=90)
        if p_id != 3:
            plt.xlabel(None)
            plt.xticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(cwd, '..', f'{dir_prefix}-result-figs',
                                 f'fewshot-param{param_set}-{dist_type}-final-{param_names[p_id]}.pdf'),
                    bbox_inches='tight', dpi=500)
        plt.close()


def read_losses(run_ids, param_set=0, fewshot=True, oneshot=False, save_meanstd=False, realoneshot=True, agent_id=0):
    """generate mean and deviation data"""
    assert len(run_ids) > 0
    if fewshot:
        assert not oneshot
        dir_prefix = f'optimisation-fewshot-param{param_set}'
    elif oneshot:
        dir_prefix = f'optimisation-oneshot-param{param_set}'
    elif realoneshot:
        agents = ['rectangle', 'round', 'cylinder']
        agent = agents[agent_id]
        dir_prefix = f'optimisation-realoneshot-{agent}-param{param_set}'
    else:
        dir_prefix = f'optimisation-param{param_set}'
    for run_id in run_ids:
        run_dir = os.path.join(cwd, '..', f'{dir_prefix}-run{run_id}-logs', )
        save_data_dir = os.path.join(run_dir, 'data')
        os.makedirs(save_data_dir, exist_ok=True)

        data_dict = {
            'seed-0': {
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
            },
            'seed-1': {
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
            },
            'seed-2': {
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
        }

        seeds = [0, 1, 2]
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


def plot_legends():
    legends = [
        'PCD CD',
        'PRT CD',
        'PCD EMD',
        'PRT EMD',
    ]

    plot.smoothed_plot_mean_deviation(
        file=os.path.join(cwd, '..', 'result-figs', 'legend.pdf'),
        legend_file=os.path.join(cwd, '..', 'result-figs', 'legend.pdf'),
        horizontal_lines=None, linestyle='--', linewidth=3, handlelength=2,
        legend=legends, legend_ncol=4, legend_frame=False,
        legend_bbox_to_anchor=(1.6, 1.1),
        legend_loc='upper right',
        data_dict_list=[None for _ in range(len(legends))], legend_only=True)


def plot_losses(param_set=0):
    loss_types = ['chamfer_loss_pcd',
                  'chamfer_loss_particle',
                  'emd_point_distance_loss',
                  'emd_particle_distance_loss',
                  'height_map_loss_pcd'
                  ]
    params = ['E', 'nu', 'yield_stress', 'rho', 'mf', 'gf']
    if param_set == 0:
        params = ['E', 'nu', 'yield_stress', 'rho']
    plt.rcParams.update({'font.size': 40})
    result_dir = os.path.join(cwd, '..', 'result-figs')
    os.makedirs(result_dir, exist_ok=True)

    fig, axes = plt.subplots(5, 5, figsize=(5 * 5, 5 * 4))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    datasets = ['fewshot', 'oneshot', 'realoneshot-rectangle', 'realoneshot-round', 'realoneshot-cylinder']
    dataset_name = ['12-mix', '6-mix', '1-rec.', '1-round', '1-cyl.']
    case = 'validation'
    for dataset_id in range(5):
        dataset = datasets[dataset_id]
        dir_prefix = f'optimisation-{dataset}-param{param_set}'
        if dataset == 'fewshot':
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
                if param_set == 1:
                    ylim_valid = [16000, 19800]
            elif loss_type == 'chamfer_loss_particle':
                y_label = 'PRT\nCD'
                ylim_valid = [5880, 6480]
                if param_set == 1:
                    ylim_valid = [10250, 15100]
            elif loss_type == 'emd_point_distance_loss':
                y_label = 'PCD\nEMD'
                ylim_valid = [1170, 1390]
                if param_set == 1:
                    ylim_valid = [1670, 2800]
            elif loss_type == 'emd_particle_distance_loss':
                y_label = 'PRT\nEMD'
                ylim_valid = [5240, 7200]
                if param_set == 1:
                    ylim_valid = [10750, 26000]
            elif loss_type == 'height_map_loss_pcd':
                y_label = 'Height\nMap'
                ylim_valid = [1970, 2400]
                if param_set == 1:
                    ylim_valid = [2500, 4000]
            else:
                raise ValueError('Unknown loss type')

            yticks = (round(ylim_valid[0] * 1.03),
                      # round((ylim_valid[1] + ylim_valid[0]) / 2),
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
                run_dir = os.path.join(cwd, '..', f'{dir_prefix}-run{run_id}-logs', 'data')
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
    plt.savefig(os.path.join(result_dir, f'loss-curves-param{param_set}.pdf'),
                bbox_inches='tight', pad_inches=0, dpi=500)
    plt.close(fig)

    fig, axes = plt.subplots(len(params), 5, figsize=(5 * 5, len(params) * 4))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    for dataset_id in range(5):
        dataset = datasets[dataset_id]
        dir_prefix = f'optimisation-{dataset}-param{param_set}'
        if dataset == 'fewshot':
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
                run_dir = os.path.join(cwd, '..', f'{dir_prefix}-run{run_id}-logs', 'data')
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

    plt.savefig(os.path.join(result_dir, f'param-curves-param{param_set}.pdf'),
                bbox_inches='tight', pad_inches=0, dpi=500)
    plt.close(fig)

def print_losses(run_ids, param_set=0, case='validation',
                 fewshot=True, oneshot=False,
                 realoneshot=True, agent_id=0):
    assert len(run_ids) > 0
    params = ['E', 'nu', 'yield_stress', 'rho', 'gf', 'mf']
    if param_set == 0:
        params = ['E', 'nu', 'yield_stress', 'rho']
    if fewshot:
        assert not oneshot
        dir_prefix = f'optimisation-fewshot-param{param_set}'
    elif oneshot:
        dir_prefix = f'optimisation-oneshot-param{param_set}'
    elif realoneshot:
        agents = ['rectangle', 'round', 'cylinder']
        agent = agents[agent_id]
        dir_prefix = f'optimisation-realoneshot-{agent}-param{param_set}'
    else:
        dir_prefix = f'optimisation-param{param_set}'

    print(f'Height map losses from {dir_prefix}')
    data_dict = dict()
    for run_id in run_ids:
        run_dir = os.path.join(cwd, '..', f'{dir_prefix}-run{run_id}-logs', 'data')
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

    return data_dict

plot_legends()

# d1 = print_losses(run_ids=[0, 1, 2, 3], param_set=1, case='validation', fewshot=True, oneshot=False, realoneshot=False, agent_id=0)
# d2 = print_losses(run_ids=[0, 1, 2, 3], param_set=1, case='validation', fewshot=False, oneshot=True, realoneshot=False, agent_id=0)
# d3 = print_losses(run_ids=[0, 1, 2, 3], param_set=1, case='validation', fewshot=False, oneshot=False, realoneshot=True, agent_id=0)
# d4 = print_losses(run_ids=[0, 1, 2, 3], param_set=1, case='validation', fewshot=False, oneshot=False, realoneshot=True, agent_id=1)
# d5 = print_losses(run_ids=[0, 1, 2, 3], param_set=1, case='validation', fewshot=False, oneshot=False, realoneshot=True, agent_id=2)
#
# print('& %.2f' %d1['run0']['min_mean_hm'], ' & %.2f' %d2['run0']['min_mean_hm'], ' & %.2f' %d3['run0']['min_mean_hm'], ' & %.2f' %d4['run0']['min_mean_hm'], ' & %.2f' %d5['run0']['min_mean_hm'], ' \\\\')
# print('& %.0f' %d1['run0']['best_p_values'][0], ' & %.0f' %d2['run0']['best_p_values'][0], ' & %.0f' %d3['run0']['best_p_values'][0], ' & %.0f' %d4['run0']['best_p_values'][0], ' & %.0f' %d5['run0']['best_p_values'][0], ' \\\\')
# print('& %.3f' %d1['run0']['best_p_values'][1], ' & %.3f' %d2['run0']['best_p_values'][1], ' & %.3f' %d3['run0']['best_p_values'][1], ' & %.3f' %d4['run0']['best_p_values'][1], ' & %.3f' %d5['run0']['best_p_values'][1], ' \\\\')
# print('& %.0f' %d1['run0']['best_p_values'][2], ' & %.0f' %d2['run0']['best_p_values'][2], ' & %.0f' %d3['run0']['best_p_values'][2], ' & %.0f' %d4['run0']['best_p_values'][2], ' & %.0f' %d5['run0']['best_p_values'][2], ' \\\\')
# print('& %.0f' %d1['run0']['best_p_values'][3], ' & %.0f' %d2['run0']['best_p_values'][3], ' & %.0f' %d3['run0']['best_p_values'][3], ' & %.0f' %d4['run0']['best_p_values'][3], ' & %.0f' %d5['run0']['best_p_values'][3], ' \\\\')
# print('& %.3f' %d1['run0']['best_p_values'][4], ' & %.3f' %d2['run0']['best_p_values'][4], ' & %.3f' %d3['run0']['best_p_values'][4], ' & %.3f' %d4['run0']['best_p_values'][4], ' & %.3f' %d5['run0']['best_p_values'][4], ' \\\\')
# print('& %.3f' %d1['run0']['best_p_values'][5], ' & %.3f' %d2['run0']['best_p_values'][5], ' & %.3f' %d3['run0']['best_p_values'][5], ' & %.3f' %d4['run0']['best_p_values'][5], ' & %.3f' %d5['run0']['best_p_values'][5], ' \\\\')
#
# print('& %.2f' %d1['run1']['min_mean_hm'], ' & %.2f' %d2['run1']['min_mean_hm'], ' & %.2f' %d3['run1']['min_mean_hm'], ' & %.2f' %d4['run1']['min_mean_hm'], ' & %.2f' %d5['run1']['min_mean_hm'], ' \\\\')
# print('& %.0f' %d1['run1']['best_p_values'][0], ' & %.0f' %d2['run1']['best_p_values'][0], ' & %.0f' %d3['run1']['best_p_values'][0], ' & %.0f' %d4['run1']['best_p_values'][0], ' & %.0f' %d5['run1']['best_p_values'][0], ' \\\\')
# print('& %.3f' %d1['run1']['best_p_values'][1], ' & %.3f' %d2['run1']['best_p_values'][1], ' & %.3f' %d3['run1']['best_p_values'][1], ' & %.3f' %d4['run1']['best_p_values'][1], ' & %.3f' %d5['run1']['best_p_values'][1], ' \\\\')
# print('& %.0f' %d1['run1']['best_p_values'][2], ' & %.0f' %d2['run1']['best_p_values'][2], ' & %.0f' %d3['run1']['best_p_values'][2], ' & %.0f' %d4['run1']['best_p_values'][2], ' & %.0f' %d5['run1']['best_p_values'][2], ' \\\\')
# print('& %.0f' %d1['run1']['best_p_values'][3], ' & %.0f' %d2['run1']['best_p_values'][3], ' & %.0f' %d3['run1']['best_p_values'][3], ' & %.0f' %d4['run1']['best_p_values'][3], ' & %.0f' %d5['run1']['best_p_values'][3], ' \\\\')
# print('& %.3f' %d1['run1']['best_p_values'][4], ' & %.3f' %d2['run1']['best_p_values'][4], ' & %.3f' %d3['run1']['best_p_values'][4], ' & %.3f' %d4['run1']['best_p_values'][4], ' & %.3f' %d5['run1']['best_p_values'][4], ' \\\\')
# print('& %.3f' %d1['run1']['best_p_values'][5], ' & %.3f' %d2['run1']['best_p_values'][5], ' & %.3f' %d3['run1']['best_p_values'][5], ' & %.3f' %d4['run1']['best_p_values'][5], ' & %.3f' %d5['run1']['best_p_values'][5], ' \\\\')
#
# print('& %.2f' %d1['run2']['min_mean_hm'], ' & %.2f' %d2['run2']['min_mean_hm'], ' & %.2f' %d3['run2']['min_mean_hm'], ' & %.2f' %d4['run2']['min_mean_hm'], ' & %.2f' %d5['run2']['min_mean_hm'], ' \\\\')
# print('& %.0f' %d1['run2']['best_p_values'][0], ' & %.0f' %d2['run2']['best_p_values'][0], ' & %.0f' %d3['run2']['best_p_values'][0], ' & %.0f' %d4['run2']['best_p_values'][0], ' & %.0f' %d5['run2']['best_p_values'][0], ' \\\\')
# print('& %.3f' %d1['run2']['best_p_values'][1], ' & %.3f' %d2['run2']['best_p_values'][1], ' & %.3f' %d3['run2']['best_p_values'][1], ' & %.3f' %d4['run2']['best_p_values'][1], ' & %.3f' %d5['run2']['best_p_values'][1], ' \\\\')
# print('& %.0f' %d1['run2']['best_p_values'][2], ' & %.0f' %d2['run2']['best_p_values'][2], ' & %.0f' %d3['run2']['best_p_values'][2], ' & %.0f' %d4['run2']['best_p_values'][2], ' & %.0f' %d5['run2']['best_p_values'][2], ' \\\\')
# print('& %.0f' %d1['run2']['best_p_values'][3], ' & %.0f' %d2['run2']['best_p_values'][3], ' & %.0f' %d3['run2']['best_p_values'][3], ' & %.0f' %d4['run2']['best_p_values'][3], ' & %.0f' %d5['run2']['best_p_values'][3], ' \\\\')
# print('& %.3f' %d1['run2']['best_p_values'][4], ' & %.3f' %d2['run2']['best_p_values'][4], ' & %.3f' %d3['run2']['best_p_values'][4], ' & %.3f' %d4['run2']['best_p_values'][4], ' & %.3f' %d5['run2']['best_p_values'][4], ' \\\\')
# print('& %.3f' %d1['run2']['best_p_values'][5], ' & %.3f' %d2['run2']['best_p_values'][5], ' & %.3f' %d3['run2']['best_p_values'][5], ' & %.3f' %d4['run2']['best_p_values'][5], ' & %.3f' %d5['run2']['best_p_values'][5], ' \\\\')
#
# print('& %.2f' %d1['run3']['min_mean_hm'], ' & %.2f' %d2['run3']['min_mean_hm'], ' & %.2f' %d3['run3']['min_mean_hm'], ' & %.2f' %d4['run3']['min_mean_hm'], ' & %.2f' %d5['run3']['min_mean_hm'], ' \\\\')
# print('& %.0f' %d1['run3']['best_p_values'][0], ' & %.0f' %d2['run3']['best_p_values'][0], ' & %.0f' %d3['run3']['best_p_values'][0], ' & %.0f' %d4['run3']['best_p_values'][0], ' & %.0f' %d5['run3']['best_p_values'][0], ' \\\\')
# print('& %.3f' %d1['run3']['best_p_values'][1], ' & %.3f' %d2['run3']['best_p_values'][1], ' & %.3f' %d3['run3']['best_p_values'][1], ' & %.3f' %d4['run3']['best_p_values'][1], ' & %.3f' %d5['run3']['best_p_values'][1], ' \\\\')
# print('& %.0f' %d1['run3']['best_p_values'][2], ' & %.0f' %d2['run3']['best_p_values'][2], ' & %.0f' %d3['run3']['best_p_values'][2], ' & %.0f' %d4['run3']['best_p_values'][2], ' & %.0f' %d5['run3']['best_p_values'][2], ' \\\\')
# print('& %.0f' %d1['run3']['best_p_values'][3], ' & %.0f' %d2['run3']['best_p_values'][3], ' & %.0f' %d3['run3']['best_p_values'][3], ' & %.0f' %d4['run3']['best_p_values'][3], ' & %.0f' %d5['run3']['best_p_values'][3], ' \\\\')
# print('& %.3f' %d1['run3']['best_p_values'][4], ' & %.3f' %d2['run3']['best_p_values'][4], ' & %.3f' %d3['run3']['best_p_values'][4], ' & %.3f' %d4['run3']['best_p_values'][4], ' & %.3f' %d5['run3']['best_p_values'][4], ' \\\\')
# print('& %.3f' %d1['run3']['best_p_values'][5], ' & %.3f' %d2['run3']['best_p_values'][5], ' & %.3f' %d3['run3']['best_p_values'][5], ' & %.3f' %d4['run3']['best_p_values'][5], ' & %.3f' %d5['run3']['best_p_values'][5], ' \\\\')


# def print_validation_losses(run_ids, param_set=1,
#                  fewshot=True, oneshot=False,
#                  realoneshot=True, agent_id=0):
#     if fewshot:
#         assert not oneshot
#         dir_prefix = f'optimisation-fewshot-param{param_set}'
#     elif oneshot:
#         dir_prefix = f'optimisation-oneshot-param{param_set}'
#     elif realoneshot:
#         agents = ['rectangle', 'round', 'cylinder']
#         agent = agents[agent_id]
#         dir_prefix = f'optimisation-realoneshot-{agent}-param{param_set}'
#     else:
#         dir_prefix = f'optimisation-param{param_set}'
#
#     print(f'Height map losses from {dir_prefix}')
#     data_dict = {
#         'rectangle-motion': {},
#         'round-motion': {},
#         'cylinder-motion': {}
#     }
#     for run_id in run_ids:
#         run_dir = os.path.join(cwd, '..', f'{dir_prefix}-result-figs', f'run{run_id}')
#         for seed in [0, 1, 2]:
#             seed_dir = os.path.join(run_dir, f'seed{seed}')
#             if os.path.isdir(seed_dir):
#                 break
#         for motion_agent in ['rectangle', 'round', 'cylinder']:
#             data_dir_0 = os.path.join(seed_dir, f'validation_tr_imgs-long_motion-{motion_agent}', '0')
#             losses_0 = json.load(open(os.path.join(data_dir_0, 'loss_info.json'), 'rb'))
#             hm_loss_0 = losses_0['height_map_loss_pcd']
#             data_dir_1 = os.path.join(seed_dir, f'validation_tr_imgs-long_motion-{motion_agent}', '1')
#             losses_1 = json.load(open(os.path.join(data_dir_1, 'loss_info.json'), 'rb'))
#             hm_loss_1 = losses_1['height_map_loss_pcd']
#             data_dict[f'{motion_agent}-motion'].update({
#                     f'data-0-run{run_id}': hm_loss_0,
#                     f'data-1-run{run_id}': hm_loss_1
#             })
#
#     return data_dict
#
# d1 = print_validation_losses(run_ids=[0, 1, 2, 3], fewshot=True, oneshot=False, realoneshot=False, agent_id=0)
# d2 = print_validation_losses(run_ids=[0, 1, 2, 3], fewshot=False, oneshot=True, realoneshot=False, agent_id=0)
# d3 = print_validation_losses(run_ids=[0, 1, 2, 3], fewshot=False, oneshot=False, realoneshot=True, agent_id=0)
# d4 = print_validation_losses(run_ids=[0, 1, 2, 3], fewshot=False, oneshot=False, realoneshot=True, agent_id=1)
# d5 = print_validation_losses(run_ids=[0, 1, 2, 3], fewshot=False, oneshot=False, realoneshot=True, agent_id=2)
#
# print('Rectangle motion')
# print('& %.2f' %d1['rectangle-motion']['data-0-run0'], '& %.2f' %d2['rectangle-motion']['data-0-run0'], '& %.2f' %d3['rectangle-motion']['data-0-run0'], '& %.2f' %d4['rectangle-motion']['data-0-run0'], '& %.2f' %d5['rectangle-motion']['data-0-run0'],' \\\\')
# print('& %.2f' %d1['rectangle-motion']['data-1-run0'], '& %.2f' %d2['rectangle-motion']['data-1-run0'], '& %.2f' %d3['rectangle-motion']['data-1-run0'], '& %.2f' %d4['rectangle-motion']['data-1-run0'], '& %.2f' %d5['rectangle-motion']['data-1-run0'],' \\\\')
# print('& %.2f' %d1['rectangle-motion']['data-0-run1'], '& %.2f' %d2['rectangle-motion']['data-0-run1'], '& %.2f' %d3['rectangle-motion']['data-0-run1'], '& %.2f' %d4['rectangle-motion']['data-0-run1'], '& %.2f' %d5['rectangle-motion']['data-0-run1'],' \\\\')
# print('& %.2f' %d1['rectangle-motion']['data-1-run1'], '& %.2f' %d2['rectangle-motion']['data-1-run1'], '& %.2f' %d3['rectangle-motion']['data-1-run1'], '& %.2f' %d4['rectangle-motion']['data-1-run1'], '& %.2f' %d5['rectangle-motion']['data-1-run1'],' \\\\')
# print('& %.2f' %d1['rectangle-motion']['data-0-run2'], '& %.2f' %d2['rectangle-motion']['data-0-run2'], '& %.2f' %d3['rectangle-motion']['data-0-run2'], '& %.2f' %d4['rectangle-motion']['data-0-run2'], '& %.2f' %d5['rectangle-motion']['data-0-run2'],' \\\\')
# print('& %.2f' %d1['rectangle-motion']['data-1-run2'], '& %.2f' %d2['rectangle-motion']['data-1-run2'], '& %.2f' %d3['rectangle-motion']['data-1-run2'], '& %.2f' %d4['rectangle-motion']['data-1-run2'], '& %.2f' %d5['rectangle-motion']['data-1-run2'],' \\\\')
# print('& %.2f' %d1['rectangle-motion']['data-0-run3'], '& %.2f' %d2['rectangle-motion']['data-0-run3'], '& %.2f' %d3['rectangle-motion']['data-0-run3'], '& %.2f' %d4['rectangle-motion']['data-0-run3'], '& %.2f' %d5['rectangle-motion']['data-0-run3'],' \\\\')
# print('& %.2f' %d1['rectangle-motion']['data-1-run3'], '& %.2f' %d2['rectangle-motion']['data-1-run3'], '& %.2f' %d3['rectangle-motion']['data-1-run3'], '& %.2f' %d4['rectangle-motion']['data-1-run3'], '& %.2f' %d5['rectangle-motion']['data-1-run3'],' \\\\')
#
# print('Round motion')
# print('& %.2f' %d1['round-motion']['data-0-run0'], '& %.2f' %d2['round-motion']['data-0-run0'], '& %.2f' %d3['round-motion']['data-0-run0'], '& %.2f' %d4['round-motion']['data-0-run0'], '& %.2f' %d5['round-motion']['data-0-run0'],' \\\\')
# print('& %.2f' %d1['round-motion']['data-1-run0'], '& %.2f' %d2['round-motion']['data-1-run0'], '& %.2f' %d3['round-motion']['data-1-run0'], '& %.2f' %d4['round-motion']['data-1-run0'], '& %.2f' %d5['round-motion']['data-1-run0'],' \\\\')
# print('& %.2f' %d1['round-motion']['data-0-run1'], '& %.2f' %d2['round-motion']['data-0-run1'], '& %.2f' %d3['round-motion']['data-0-run1'], '& %.2f' %d4['round-motion']['data-0-run1'], '& %.2f' %d5['round-motion']['data-0-run1'],' \\\\')
# print('& %.2f' %d1['round-motion']['data-1-run1'], '& %.2f' %d2['round-motion']['data-1-run1'], '& %.2f' %d3['round-motion']['data-1-run1'], '& %.2f' %d4['round-motion']['data-1-run1'], '& %.2f' %d5['round-motion']['data-1-run1'],' \\\\')
# print('& %.2f' %d1['round-motion']['data-0-run2'], '& %.2f' %d2['round-motion']['data-0-run2'], '& %.2f' %d3['round-motion']['data-0-run2'], '& %.2f' %d4['round-motion']['data-0-run2'], '& %.2f' %d5['round-motion']['data-0-run2'],' \\\\')
# print('& %.2f' %d1['round-motion']['data-1-run2'], '& %.2f' %d2['round-motion']['data-1-run2'], '& %.2f' %d3['round-motion']['data-1-run2'], '& %.2f' %d4['round-motion']['data-1-run2'], '& %.2f' %d5['round-motion']['data-1-run2'],' \\\\')
# print('& %.2f' %d1['round-motion']['data-0-run3'], '& %.2f' %d2['round-motion']['data-0-run3'], '& %.2f' %d3['round-motion']['data-0-run3'], '& %.2f' %d4['round-motion']['data-0-run3'], '& %.2f' %d5['round-motion']['data-0-run3'],' \\\\')
# print('& %.2f' %d1['round-motion']['data-1-run3'], '& %.2f' %d2['round-motion']['data-1-run3'], '& %.2f' %d3['round-motion']['data-1-run3'], '& %.2f' %d4['round-motion']['data-1-run3'], '& %.2f' %d5['round-motion']['data-1-run3'],' \\\\')
#
# print('Cylinder motion')
# print('& %.2f' %d1['cylinder-motion']['data-0-run0'], '& %.2f' %d2['cylinder-motion']['data-0-run0'], '& %.2f' %d3['cylinder-motion']['data-0-run0'], '& %.2f' %d4['cylinder-motion']['data-0-run0'], '& %.2f' %d5['cylinder-motion']['data-0-run0'],' \\\\')
# print('& %.2f' %d1['cylinder-motion']['data-1-run0'], '& %.2f' %d2['cylinder-motion']['data-1-run0'], '& %.2f' %d3['cylinder-motion']['data-1-run0'], '& %.2f' %d4['cylinder-motion']['data-1-run0'], '& %.2f' %d5['cylinder-motion']['data-1-run0'],' \\\\')
# print('& %.2f' %d1['cylinder-motion']['data-0-run1'], '& %.2f' %d2['cylinder-motion']['data-0-run1'], '& %.2f' %d3['cylinder-motion']['data-0-run1'], '& %.2f' %d4['cylinder-motion']['data-0-run1'], '& %.2f' %d5['cylinder-motion']['data-0-run1'],' \\\\')
# print('& %.2f' %d1['cylinder-motion']['data-1-run1'], '& %.2f' %d2['cylinder-motion']['data-1-run1'], '& %.2f' %d3['cylinder-motion']['data-1-run1'], '& %.2f' %d4['cylinder-motion']['data-1-run1'], '& %.2f' %d5['cylinder-motion']['data-1-run1'],' \\\\')
# print('& %.2f' %d1['cylinder-motion']['data-0-run2'], '& %.2f' %d2['cylinder-motion']['data-0-run2'], '& %.2f' %d3['cylinder-motion']['data-0-run2'], '& %.2f' %d4['cylinder-motion']['data-0-run2'], '& %.2f' %d5['cylinder-motion']['data-0-run2'],' \\\\')
# print('& %.2f' %d1['cylinder-motion']['data-1-run2'], '& %.2f' %d2['cylinder-motion']['data-1-run2'], '& %.2f' %d3['cylinder-motion']['data-1-run2'], '& %.2f' %d4['cylinder-motion']['data-1-run2'], '& %.2f' %d5['cylinder-motion']['data-1-run2'],' \\\\')
# print('& %.2f' %d1['cylinder-motion']['data-0-run3'], '& %.2f' %d2['cylinder-motion']['data-0-run3'], '& %.2f' %d3['cylinder-motion']['data-0-run3'], '& %.2f' %d4['cylinder-motion']['data-0-run3'], '& %.2f' %d5['cylinder-motion']['data-0-run3'],' \\\\')
# print('& %.2f' %d1['cylinder-motion']['data-1-run3'], '& %.2f' %d2['cylinder-motion']['data-1-run3'], '& %.2f' %d3['cylinder-motion']['data-1-run3'], '& %.2f' %d4['cylinder-motion']['data-1-run3'], '& %.2f' %d5['cylinder-motion']['data-1-run3'],' \\\\')