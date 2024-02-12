import os
import json
import numpy as np
from drl_implementation.agent.utils import plot as plot
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator


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

"""generate mean and deviation data"""

run_id = 0
for run_id in [3]:
    run_dir = os.path.join(cwd, '..', f'optimisation-fewshot-param0-run{run_id}-logs',)

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
                            if v.tag[6:] == 'point_distance_sr':
                                data_dict[f'seed-{seed}']['training']['point_distance_sr'].append(v.simple_value)
                            elif v.tag[6:]
                        print(v.tag, event.step, v.simple_value)
    for loss_type in loss_types:
        # training
        data = []
        for seed in seeds:
            with open(os.path.join(run_dir, f'run-seed-{str(seed)}-tag-Loss_{loss_type}.json'), 'rb') as f:
                d = json.load(f)
            data.append(np.array(d)[:, -1])
        plot.get_mean_and_deviation(np.array(data),
                                    save_data=True,
                                    file_name=os.path.join(run_dir, f'training-{loss_type}.json'))
        # validation
        validation_data = []
        for seed in seeds:
            with open(os.path.join(run_dir, f'run-seed-{str(seed)}-tag-Validation loss_{loss_type}.json'), 'rb') as f:
                d = json.load(f)
            validation_data.append(np.array(d)[:, -1])
        plot.get_mean_and_deviation(np.array(validation_data),
                                    save_data=True,
                                    file_name=os.path.join(run_dir, f'validation-{loss_type}.json'))
exit()

"""Plotting"""
for case in ['training', 'validation']:
    for i in range(len(loss_types)):
        loss_type = loss_types[i]
        stat_dicts = []
        for run_id in [0, 1]:
            run_dir = os.path.join(cwd, '..', f'optimisation-fewshot-param0-run{run_id}-logs', 'data')

            with open(os.path.join(run_dir, f'{case}-{loss_type}.json'), 'rb') as f:
                d = json.load(f)
            if i <= 2:
                d['mean'] = np.array(d['mean']).tolist()
                d['lower'] = np.array(d['lower']).tolist()
                d['upper'] = np.array(d['upper']).tolist()
            stat_dicts.append(d)

        plot.smoothed_plot_mean_deviation(
            file=os.path.join(cwd, '..', 'optimisation-result-figs', f'fewshot-param0-{case}-{loss_type}.pdf'),
            data_dict_list=stat_dicts,
            horizontal_lines=None, linestyle='--', linewidth=5,
            legend=None, legend_ncol=2, legend_frame=False,
            legend_bbox_to_anchor=(-0.1, 1.0),
            legend_loc='upper left',
            x_label='Epoch', x_axis_off=False,
            y_label=None, y_axis_off=False,
            title=loss_type
        )