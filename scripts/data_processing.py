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


def read_plot_params(run_ids, dist_type='Euclidean', fewshot=True):
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
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan','k']
    param_names = ['E', 'nu', 'yield stress', 'rho', 'mf', 'gf']
    if fewshot:
        dir_prefix = 'optimisation-fewshot'
    else:
        dir_prefix = 'optimisation'
    for p_id in range(4):
        n = 0
        if p_id != 3:
            plt.figure(figsize=(8, 2.5))
        else:
            plt.figure(figsize=(8, 5))
        for run_id in run_ids:
            run_dir = os.path.join(cwd, '..', f'{dir_prefix}-param0-run{run_id}-logs',)
            values = []
            for seed in [0, 1, 2]:
                seed_dir = os.path.join(run_dir, f'seed-{str(seed)}')
                params = np.load(os.path.join(seed_dir, 'final_params.npy'))
                plt.scatter(n, params.flatten()[p_id], color=colors[run_id])
                n += 1
        plt.title(f'Final value of {param_names[p_id]}')
        plt.xticks(np.arange(15), legends, rotation=90)
        if p_id != 3:
            plt.xlabel(None)
            plt.xticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(cwd, '..', f'{dir_prefix}-result-figs',
                                 f'fewshot-param0-{dist_type}-final-{param_names[p_id]}.pdf'), bbox_inches='tight', dpi=500)
        plt.close()


def generate_mean_deviation(run_ids, fewshot=True):
    """generate mean and deviation data"""
    assert len(run_ids) > 0
    if fewshot:
        dir_prefix = 'optimisation-fewshot'
    else:
        dir_prefix = 'optimisation'
    for run_id in run_ids:
        run_dir = os.path.join(cwd, '..', f'{dir_prefix}-param0-run{run_id}-logs',)
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
                                    data_dict[f'seed-{seed}']['training']['chamfer_loss_particle'].append(v.simple_value)
                                elif v.tag[5:] == 'height_map_loss_pcd':
                                    data_dict[f'seed-{seed}']['training']['height_map_loss_pcd'].append(v.simple_value)
                                elif v.tag[5:] == 'emd_point_distance_loss':
                                    data_dict[f'seed-{seed}']['training']['emd_point_distance_loss'].append(v.simple_value)
                                elif v.tag[5:] == 'emd_particle_distance_loss':
                                    data_dict[f'seed-{seed}']['training']['emd_particle_distance_loss'].append(v.simple_value)
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
                                    data_dict[f'seed-{seed}']['validation']['particle_distance_sr'].append(v.simple_value)
                                elif v.tag[16:] == 'particle_distance_rs':
                                    data_dict[f'seed-{seed}']['validation']['particle_distance_rs'].append(v.simple_value)
                                elif v.tag[16:] == 'chamfer_loss_particle':
                                    data_dict[f'seed-{seed}']['validation']['chamfer_loss_particle'].append(v.simple_value)
                                elif v.tag[16:] == 'height_map_loss_pcd':
                                    data_dict[f'seed-{seed}']['validation']['height_map_loss_pcd'].append(v.simple_value)
                                elif v.tag[16:] == 'emd_point_distance_loss':
                                    data_dict[f'seed-{seed}']['validation']['emd_point_distance_loss'].append(v.simple_value)
                                elif v.tag[16:] == 'emd_particle_distance_loss':
                                    data_dict[f'seed-{seed}']['validation']['emd_particle_distance_loss'].append(v.simple_value)
                                else:
                                    pass
                            else:
                                pass

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


def plot_legends(dist_type='Euclidean', fewshot=True):
    legends = [
        f'{dist_type}: PCD chamfer',
        f'{dist_type}: Particle chamfer',
        f'{dist_type}: PCD emd',
        f'{dist_type}: Particle emd',
        f'{dist_type}: Height map',
    ]
    if fewshot:
        dir_prefix = 'optimisation-fewshot'
    else:
        dir_prefix = 'optimisation'
    plot.smoothed_plot_mean_deviation(
        file=os.path.join(cwd, '..', f'{dir_prefix}-result-figs', f'fewshot-param0-{dist_type}-legend.pdf'),
        legend_file=os.path.join(cwd, '..', f'{dir_prefix}-result-figs', f'fewshot-param0-{dist_type}-legend.pdf'),
        horizontal_lines=None, linestyle='--', linewidth=5,
        legend=legends, legend_ncol=1, legend_frame=False,
        legend_bbox_to_anchor=(1.6, 1.1),
        legend_loc='upper right',
        data_dict_list=[None for _ in range(len(legends))], legend_only=True)


def plot_losses(run_ids, dist_type='Euclidean', fewshot=True):
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
    if fewshot:
        dir_prefix = 'optimisation-fewshot'
    else:
        dir_prefix = 'optimisation'
    for case in ['training', 'validation']:
        for i in range(len(loss_types)):
            loss_type = loss_types[i]
            if loss_type == 'point_distance_sr':
                title = f'{dist_type}: PCD sim2real'
            elif loss_type == 'point_distance_rs':
                title = f'{dist_type}: PCD real2sim'
            elif loss_type == 'chamfer_loss_pcd':
                title = f'{dist_type}: PCD chamfer'
            elif loss_type == 'particle_distance_sr':
                title = f'{dist_type}: Particle sim2real'
            elif loss_type == 'particle_distance_rs':
                title = f'{dist_type}: Particle real2sim'
            elif loss_type == 'chamfer_loss_particle':
                title = f'{dist_type}: Particle chamfer'
            elif loss_type == 'height_map_loss_pcd':
                title = f'{dist_type}: Height map'
            elif loss_type == 'emd_point_distance_loss':
                title = f'{dist_type}: PCD emd'
            elif loss_type == 'emd_particle_distance_loss':
                title = f'{dist_type}: Particle emd'
            else:
                raise ValueError('Unknown loss type')
            stat_dicts = []
            for run_id in run_ids:
                run_dir = os.path.join(cwd, '..', f'{dir_prefix}-param0-run{run_id}-logs', 'data')

                with open(os.path.join(run_dir, f'{case}-{loss_type}.json'), 'rb') as f:
                    d = json.load(f)
                if i <= 2:
                    d['mean'] = np.array(d['mean']).tolist()
                    d['lower'] = np.array(d['lower']).tolist()
                    d['upper'] = np.array(d['upper']).tolist()
                stat_dicts.append(d)

            plot.smoothed_plot_mean_deviation(
                file=os.path.join(cwd, '..', f'{dir_prefix}-result-figs', f'fewshot-param0-{dist_type}-{case}-{loss_type}.pdf'),
                data_dict_list=stat_dicts,
                horizontal_lines=None, linestyle='--', linewidth=5,
                legend=None, legend_ncol=1, legend_frame=False,
                legend_bbox_to_anchor=(1.4, 1.1),
                legend_loc='upper right',
                x_label='Epoch', x_axis_off=False,
                y_label=None, y_axis_off=False,
                title=title
            )


# generate_mean_deviation([8, 9])
plot_legends(dist_type='Euclidean')
plot_losses(run_ids=[2, 3, 1, 0, 4], dist_type='Euclidean')
read_plot_params(run_ids=[2, 3, 1, 0, 4], dist_type='Euclidean')
plot_legends(dist_type='Exponential')
plot_losses(run_ids=[5, 6, 8, 9, 7], dist_type='Exponential')
read_plot_params(run_ids=[5, 6, 8, 9, 7], dist_type='Exponential')