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
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams["font.weight"] = "bold"
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
        'PCD chamfer',
        'Particle chamfer',
        'PCD emd',
        'Particle emd',
    ]

    plot.smoothed_plot_mean_deviation(
        file=os.path.join(cwd, '..', 'result-figs', 'legend.pdf'),
        legend_file=os.path.join(cwd, '..', 'result-figs', 'legend.pdf'),
        horizontal_lines=None, linestyle='--', linewidth=7, handlelength=0.5,
        legend=legends, legend_ncol=1, legend_frame=False,
        legend_bbox_to_anchor=(1.6, 1.1),
        legend_loc='upper right',
        data_dict_list=[None for _ in range(len(legends))], legend_only=True)


def plot_losses(run_ids, param_set=0, dist_type='Euclidean', fewshot=True, oneshot=False,
                mean_std=True, params=False, realoneshot=True, agent_id=0):
    plt.rcParams.update({'font.size': 32})
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
    ]
    print(f'[Warning]: make sure the run ids are in the correct order based on \n'
          f'{legends[0]}\n{legends[3]}\n{legends[6]}\n{legends[9]}')
    if fewshot:
        assert not oneshot
        dir_prefix = f'optimisation-fewshot-param{param_set}'
        file_prefix = f'fewshot-param{param_set}'
    elif oneshot:
        dir_prefix = f'optimisation-oneshot-param{param_set}'
        file_prefix = f'oneshot-param{param_set}'
    elif realoneshot:
        agents = ['rectangle', 'round', 'cylinder']
        agent = agents[agent_id]
        dir_prefix = f'optimisation-realoneshot-{agent}-param{param_set}'
        file_prefix = f'realoneshot-{agent}-param{param_set}'
    else:
        dir_prefix = f'optimisation-param{param_set}'
        file_prefix = f'param{param_set}'

    result_dir = os.path.join(cwd, '..', f'{dir_prefix}-result-figs')
    os.makedirs(result_dir, exist_ok=True)

    """Plot losses"""
    for case in ['training', 'validation']:
        for i in range(len(loss_types)):
            loss_type = loss_types[i]
            if loss_type == 'point_distance_sr':
                title = f'{dist_type}: PCD sim2real'
                ylim_valid = [7500, 8450]
                if param_set == 1:
                    ylim_valid = [14000, 17000]
            elif loss_type == 'point_distance_rs':
                title = f'{dist_type}: PCD real2sim'
                ylim_valid = [1110, 1260]
                if param_set == 1:
                    ylim_valid = [1560, 2400]
            elif loss_type == 'chamfer_loss_pcd':
                title = f'{dist_type}: PCD chamfer'
                ylim_valid = [8690, 9650]
                if param_set == 1:
                    ylim_valid = [16000, 18900]
            elif loss_type == 'particle_distance_sr':
                title = f'{dist_type}: Particle sim2real'
                ylim_valid = [2760, 3050]
                if param_set == 1:
                    ylim_valid = [4600, 6600]
            elif loss_type == 'particle_distance_rs':
                title = f'{dist_type}: Particle real2sim'
                ylim_valid = [3010, 3450]
                if param_set == 1:
                    ylim_valid = [5500, 8200]
            elif loss_type == 'chamfer_loss_particle':
                title = f'{dist_type}: Particle chamfer'
                ylim_valid = [5880, 6480]
                if param_set == 1:
                    ylim_valid = [10250, 14900]
            elif loss_type == 'height_map_loss_pcd':
                title = f'{dist_type}: Height map'
                ylim_valid = [1970, 2400]
                if param_set == 1:
                    ylim_valid = [2500, 8500]
            elif loss_type == 'emd_point_distance_loss':
                title = f'{dist_type}: PCD emd'
                ylim_valid = [1170, 1390]
                if param_set == 1:
                    ylim_valid = [1670, 2700]
            elif loss_type == 'emd_particle_distance_loss':
                title = f'{dist_type}: Particle emd'
                ylim_valid = [5240, 7200]
                if param_set == 1:
                    ylim_valid = [10750, 25000]
            else:
                raise ValueError('Unknown loss type')

            yticks = (round(ylim_valid[0] * 1.01),
                      round((ylim_valid[1] + ylim_valid[0]) / 2),
                      round(ylim_valid[1] * 0.99))

            # if mean_std:
            #     stat_dicts = []
            #     max_y = 0
            #     min_y = 2000000
            #     for run_id in run_ids:
            #         run_dir = os.path.join(cwd, '..', f'{dir_prefix}-run{run_id}-logs', 'data')
            #
            #         with open(os.path.join(run_dir, f'{case}-{loss_type}.json'), 'rb') as f:
            #             d = json.load(f)
            #
            #         max_y = np.max([max_y, np.max(d['upper'])])
            #         min_y = np.min([min_y, np.min(d['lower'])])
            #
            #         d['mean'] = np.array(d['mean']).tolist()
            #         d['lower'] = np.array(d['lower']).tolist()
            #         d['upper'] = np.array(d['upper']).tolist()
            #         stat_dicts.append(d)
            #
            #     if case == 'training':
            #         ylim_valid = (min_y * 0.995, max_y * 1.005)
            #         delta_y = (max_y - min_y) / 30
            #         yticks = (round(min_y + delta_y),
            #                   round((min_y + max_y) / 2),
            #                   round(max_y - delta_y))
            #
            #     plot.smoothed_plot_mean_deviation(
            #         file=os.path.join(cwd, '..', f'{dir_prefix}-result-figs',
            #                           f'{file_prefix}-{dist_type}-{case}-{loss_type}-meanstd.pdf'),
            #         data_dict_list=stat_dicts,
            #         horizontal_lines=None, linestyle='--', linewidth=5,
            #         legend=None, legend_ncol=1, legend_frame=False,
            #         legend_bbox_to_anchor=(1.4, 1.1),
            #         legend_loc='upper right',
            #         x_label='Epoch', x_axis_off=True,
            #         y_label=None, y_axis_off=False, ylim=ylim_valid, yticks=yticks,
            #         title=None
            #     )
            # else:
            color_pool = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'k']
            datas = []
            colors = []
            linestyles = []
            linewidths = []
            alphas = []
            n = 0
            for run_id in run_ids:
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

            max_y = np.max(datas)
            min_y = np.min(datas)
            if case == 'training':
                ylim_valid = (min_y * 0.995, max_y * 1.005)
                delta_y = (max_y - min_y) / 30
                yticks = (round(min_y + delta_y),
                          round((min_y + max_y) / 2),
                          round(max_y - delta_y))

            plot.smoothed_plot_multi_line(
                file=os.path.join(cwd, '..', f'{dir_prefix}-result-figs',
                                  f'{file_prefix}-{dist_type}-{case}-{loss_type}-raw-mean.pdf'),
                window=10,
                data=datas, colors=colors, linestyles=linestyles, linewidths=linewidths, alphas=alphas,
                x_axis_off=True,
                y_label=None, y_axis_off=False, ylim=ylim_valid, yticks=yticks
            )

    """Plot params"""
    if params:
        params = ['E', 'nu', 'yield_stress', 'rho', 'mf', 'gf']
        if param_set == 0:
            params = ['E', 'nu', 'yield_stress', 'rho']

        for p in params:
            color_pool = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'k']
            if p == 'E':
                ylim_valid = [1e4 - 10000, 3e5 + 10000]
                yticks = (1e4, 155000, 3e5)
            elif p == 'nu':
                ylim_valid = [-0.03, 0.52]
                yticks = (0, 0.25, 0.5)
            elif p == 'yield_stress':
                ylim_valid = [1e3 - 1000, 2e4 + 1000]
                yticks = (1e3, 10500, 2e4)
            elif p == 'rho':
                ylim_valid = [980, 2020]
                yticks = (1000, 1500, 2000)
            else:
                ylim_valid = [-0.05, 2.05]
                yticks = (0, 1, 2)

            datas = []
            colors = []
            n = 0
            for run_id in run_ids:
                run_dir = os.path.join(cwd, '..', f'{dir_prefix}-run{run_id}-logs', 'data')
                losses = json.load(open(os.path.join(run_dir, 'raw_loss.json'), 'rb'))
                for seed in [0, 1, 2]:
                    datas.append(losses[f'seed-{seed}']['parameters'][p])
                    colors.append(color_pool[n])
                n += 1

            plot.smoothed_plot_multi_line(
                file=os.path.join(cwd, '..', f'{dir_prefix}-result-figs',
                                  f'{file_prefix}-{dist_type}-param-{p}-raw.pdf'),
                data=datas, colors=colors,
                x_axis_off=True,
                y_label=None, y_axis_off=False, ylim=ylim_valid, yticks=yticks
            )


# read_losses(run_ids=[0, 1, 2, 3], param_set=0, fewshot=False, oneshot=False, save_meanstd=True, realoneshot=True, agent_id=0)
# read_losses(run_ids=[0, 1, 2, 3], param_set=0, fewshot=False, oneshot=False, save_meanstd=True, realoneshot=True, agent_id=1)
# read_losses(run_ids=[0, 1, 2, 3], param_set=0, fewshot=False, oneshot=False, save_meanstd=True, realoneshot=True, agent_id=2)
# plot_legends()
plot_losses(run_ids=[0, 1, 2, 3], param_set=0, dist_type='Euclidean', params=False, fewshot=False,
            oneshot=False, realoneshot=True, agent_id=0)
plot_losses(run_ids=[0, 1, 2, 3], param_set=0, dist_type='Euclidean', params=False, fewshot=False,
            oneshot=False, realoneshot=True, agent_id=1)
plot_losses(run_ids=[0, 1, 2, 3], param_set=0, dist_type='Euclidean', params=False, fewshot=False,
            oneshot=False, realoneshot=True, agent_id=2)
# plot_losses(run_ids=[0, 1, 2, 3], param_set=0, dist_type='Euclidean', params=False, fewshot=False, oneshot=True)
# plot_losses(run_ids=[0, 1, 2, 3], param_set=0, dist_type='Euclidean', params_only=True, fewshot=True, oneshot=False)
# plot_losses(run_ids=[0, 1, 2, 3], param_set=0, dist_type='Euclidean', params_only=False, fewshot=False, oneshot=True)
# plot_losses(run_ids=[0, 1, 2, 3], param_set=0, dist_type='Exponential', params=False, fewshot=False, oneshot=True)
# plot_losses(run_ids=[0, 1, 2, 3], param_set=0, dist_type='Euclidean', params_only=True, fewshot=False, oneshot=True)
# read_plot_params(run_ids=[2, 3, 1, 0, 4], param_set=0, dist_type='Euclidean')
# plot_legends(dist_type='Exponential', param_set=0)
# plot_losses(run_ids=[5, 6, 7, 8], param_set=1, dist_type='Exponential', params_only=False, fewshot=True, oneshot=False)
# plot_losses(run_ids=[5, 6, 7, 8], param_set=1, dist_type='Exponential', params_only=False, fewshot=True, oneshot=False)
# plot_losses(run_ids=[5, 6, 7, 8], param_set=1, dist_type='Exponential', params_only=True, fewshot=True, oneshot=False)
# plot_losses(run_ids=[5, 6, 7, 8], param_set=0, dist_type='Exponential', params_only=False, fewshot=False, oneshot=True)
# plot_losses(run_ids=[5, 6, 7, 8], param_set=0, dist_type='Exponential', params_only=False, fewshot=False, oneshot=True)
# plot_losses(run_ids=[5, 6, 7, 8], param_set=0, dist_type='Exponential', params_only=True, fewshot=False, oneshot=True)
# read_plot_params(run_ids=[5, 6, 8, 9, 7], param_set=0, dist_type='Exponential')
