import os
import re
script_path = os.path.dirname(os.path.realpath(__file__))
import numpy as np
from doma.engine.utils.misc import plot_loss_landscape
DTYPE_NP = np.float32

xy_param = 'rho-yieldstress'
rho_list = np.arange(1000, 1999, 33.3).astype(DTYPE_NP)
yield_stress_list = np.arange(1000, 20200, 640).astype(DTYPE_NP)
rho, yield_stress = np.meshgrid(rho_list, yield_stress_list)

point_distance_sr = np.zeros_like(rho)
point_distance_rs = np.zeros_like(rho)
chamfer_loss_pcd = np.zeros_like(rho)
particle_distance_sr = np.zeros_like(rho)
particle_distance_rs = np.zeros_like(rho)
chamfer_loss_particle = np.zeros_like(rho)
height_map_loss_pcd = np.zeros_like(rho)
emd_point_distance_loss = np.zeros_like(rho)
emd_particle_distance_loss = np.zeros_like(rho)

file_path = os.path.join(script_path, '..', 'loss-landscapes-m34-one-shot',
                         'loss_landscape_euclidean_rho-yieldstress_40000000.0pd.log')

with open(file_path, 'r') as file:
    for line in file:
        match_id = re.search(r'INFO The (\d+), (\d+)-th loss is:', line)
        if match_id:
            print('Reading loss at', match_id.group(1), match_id.group(2))
            i = int(match_id.group(1))
            j = int(match_id.group(2))
        match_v = re.search(r'INFO (\w+): (\d+\.\d+)', line)
        if match_v:
            loss_name = match_v.group(1)
            loss_value = float(match_v.group(2))
            print(loss_name, loss_value)

            if loss_name == 'point_distance_sr':
                point_distance_sr[j, i] = loss_value
            elif loss_name == 'point_distance_rs':
                point_distance_rs[j, i] = loss_value
            elif loss_name == 'chamfer_loss_pcd':
                chamfer_loss_pcd[j, i] = loss_value
            elif loss_name == 'particle_distance_sr':
                particle_distance_sr[j, i] = loss_value
            elif loss_name == 'particle_distance_rs':
                particle_distance_rs[j, i] = loss_value
            elif loss_name == 'chamfer_loss_particle':
                chamfer_loss_particle[j, i] = loss_value
            elif loss_name == 'height_map_loss_pcd':
                height_map_loss_pcd[j, i] = loss_value
            elif loss_name == 'emd_point_distance_loss':
                emd_point_distance_loss[j, i] = loss_value
            elif loss_name == 'emd_particle_distance_loss':
                emd_particle_distance_loss[j, i] = loss_value
            else:
                pass

n_datapoints = 6
losses = [point_distance_sr / n_datapoints,
          point_distance_rs / n_datapoints,
          chamfer_loss_pcd / n_datapoints,
          particle_distance_sr / n_datapoints,
          particle_distance_rs / n_datapoints,
          chamfer_loss_particle / n_datapoints,
          height_map_loss_pcd / n_datapoints,
          emd_point_distance_loss / n_datapoints,
          emd_particle_distance_loss / n_datapoints
          ]
fig_data_path = os.path.join(script_path, '..', 'loss-landscapes-m34-one-shot')
loss_types = [
    'point_distance_sr', 'point_distance_rs', 'chamfer_loss_pcd',
    'particle_distance_sr', 'particle_distance_rs', 'chamfer_loss_particle',
    'height_map_loss_pcd',
    'emd_point_distance_loss', 'emd_particle_distance_loss'
]
distance_type = 'euclidean'
for i in range(len(losses)):
    np.save(os.path.join(fig_data_path, f'{loss_types[i]}_{distance_type}_{xy_param}-40000000.0pd.npy'), losses[i])
    plot_loss_landscape(rho, yield_stress, losses[i], fig_title=None, colorbar=True, cmap='YlGnBu',
                        x_label='rho', y_label='yield_stress', z_label='Loss', hm=True, show=False, save=True,
                        path=os.path.join(fig_data_path,
                                          f"{loss_types[i]}_{distance_type}_landscape_{xy_param}-topview-40000000.0pd.pdf"))
