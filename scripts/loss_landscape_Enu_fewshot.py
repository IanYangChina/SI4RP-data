import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import numpy as np
import os
import taichi as ti
from doma.envs.sys_id_env import make_env, set_parameters
from time import time
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 12})
script_path = os.path.dirname(os.path.realpath(__file__))
fig_data_path = os.path.join(script_path, '..', 'loss-landscapes-few-shot')
os.makedirs(fig_data_path, exist_ok=True)
DTYPE_NP = np.float32
DTYPE_TI = ti.f32


def plot_loss_landscape(p1, p2, loss, fig_title='Fig', loss_type='bi_chamfer_loss', file_suffix='', view='left',
                        x_label='p1', y_label='p2', z_label='Loss', hm=False, show=False, save=True):
    cmap = 'GnBu'
    if not show:
        mpl.use('Agg')
    if hm:
        loss = np.flip(loss, axis=0)
        fig, ax = plt.subplots()
        im = ax.imshow(loss, cmap=cmap)
        fig.colorbar(im, ax=ax)
        x_ticks = np.linspace(0, p1.shape[1] - 1, 5).astype(np.int32)
        x_labels = np.round(np.linspace(p1[0, 0], p1[0, -1], 5), 2)

        y_ticks = np.linspace(0, p1.shape[0] - 1, 5).astype(np.int32)
        y_labels = np.round(np.linspace(p2[0, 0], p2[-1, 0], 5), 2).tolist()
        y_labels.reverse()
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(label=fig_title)
        if save:
            plt.savefig(os.path.join(fig_data_path, f"{loss_type}_landscape{file_suffix}.pdf"), dpi=500, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    else:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        surf = ax.plot_surface(p1, p2, loss, cmap=cmap,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.zaxis.set_major_locator(LinearLocator(5))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')
        if view == 'left':
            ax.view_init(elev=25., azim=130)
        else:
            ax.view_init(elev=25., azim=-130)
        ax.tick_params(axis='x', pad=0)
        ax.tick_params(axis='y', pad=0)
        ax.tick_params(axis='z', pad=10)
        ax.set_xlabel(x_label)
        ax.xaxis.labelpad = 5
        ax.set_ylabel(y_label)
        ax.yaxis.labelpad = 5
        ax.set_zlabel(z_label)
        ax.zaxis.labelpad = 22
        ax.set_title(label=fig_title)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5, location=view)
        if save:
            plt.savefig(os.path.join(fig_data_path, f"{loss_type}_landscape{file_suffix}.pdf"), dpi=500, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()


p_density = 4e7
p_density_str = '4e7pd'

loss_cfg = {
    'exponential_distance': False,
    'averaging_loss': False,
    'point_distance_rs_loss': True,
    'point_distance_sr_loss': False,
    'down_sample_voxel_size': 0.005,
    'particle_distance_rs_loss': False,
    'particle_distance_sr_loss': True,
    'voxelise_res': 1080,
    'ptcl_density': p_density,
    'load_height_map': True,
    'height_map_loss': True,
    'height_map_res': 32,
    'height_map_size': 0.11,
    'emd_point_distance_loss': False,
    'emd_particle_distance_loss': False,
}

xy_param = 'E-nu'
E_list = np.arange(1e4, 3e5, 10000).astype(DTYPE_NP)
nu_list = np.arange(0.01, 0.49, 0.016).astype(DTYPE_NP)

E, nu = np.meshgrid(E_list, nu_list)
yield_stress = 1e5
rho = 1300.0

distance_type = 'euclidean'

point_distance_sr = np.zeros_like(E)
point_distance_rs = np.zeros_like(E)
chamfer_loss_pcd = np.zeros_like(E)
particle_distance_sr = np.zeros_like(E)
particle_distance_rs = np.zeros_like(E)
chamfer_loss_particle = np.zeros_like(E)
height_map_loss_pcd = np.zeros_like(E)
emd_point_distance_loss = np.zeros_like(E)
emd_particle_distance_loss = np.zeros_like(E)

n_datapoints = 12
data_id_dict = {
    '1': {'rectangle': [3, 5], 'round': [0, 1], 'cylinder': [1, 2]},
    '2': {'rectangle': [1, 3], 'round': [0, 2], 'cylinder': [0, 2]},
}
# Load trajectories.
for motion_ind in ['1', '2']:
    dt_global = 0.01
    trajectory = np.load(os.path.join(script_path, '..', 'trajectories', f'tr_{motion_ind}_v_dt_{dt_global:0.2f}.npy'))
    horizon = trajectory.shape[0]
    n_substeps = 50

    for agent in ['rectangle', 'round', 'cylinder']:
        training_data_path = os.path.join(script_path, '..', f'data-motion-{motion_ind}', f'eef-{agent}')
        if agent == 'rectangle':
            agent_init_euler = (0, 0, 45)
        else:
            agent_init_euler = (0, 0, 0)
        data_ids = data_id_dict[motion_ind][agent]
        for data_ind in data_ids:
            ti.reset()
            ti.init(arch=ti.opengl, default_ip=ti.i32, default_fp=DTYPE_TI, fast_math=False, random_seed=1)
            data_cfg = {
                'data_path': training_data_path,
                'data_ind': str(data_ind),
            }
            env_cfg = {
                'p_density': p_density,
                'horizon': horizon,
                'dt_global': dt_global,
                'n_substeps': n_substeps,
                'material_id': 2,
                'agent_name': agent,
                'agent_init_euler': agent_init_euler,
            }

            env, mpm_env, init_state = make_env(data_cfg, env_cfg, loss_cfg)
            print(f'===> Num. simulation particles: {mpm_env.loss.n_particles_matching_mat}')
            print(f'===> Num. target pcd points: {mpm_env.loss.n_target_pcd_points}')
            print(f'===> Num. target particles: {mpm_env.loss.n_target_particles_from_mesh}')
            t0 = time()
            print(f'Start calculating losses with grid size: {point_distance_sr.shape}')
            for i in range(len(E_list)):
                for j in range(len(nu_list)):
                    set_parameters(mpm_env, env_cfg['material_id'],  E_list[i], nu_list[j],
                                   yield_stress=yield_stress, rho=rho,
                                   manipulator_friction=0.2, ground_friction=2.0)
                    mpm_env.set_state(init_state['state'], grad_enabled=False)
                    for k in range(mpm_env.horizon):
                        action = trajectory[k].copy()
                        mpm_env.step(action)
                    loss_info = mpm_env.get_final_loss()

                    abort = False
                    print(f'The {i}, {j}-th loss is:')
                    for b, v in loss_info.items():
                        if b == 'final_height_map':
                            continue
                        # Check if the loss is strange
                        if np.isinf(v) or np.isnan(v):
                            abort = True
                        print(f'{b}: {v:.4f}')

                    if abort:
                        print(f'===> [Warning] Strange loss.')
                        print(f'===> [Warning] E: {E_list[i]}, nu: {nu_list[j]}')
                        print(f'===> [Warning] Motion: {motion_ind}, agent: {agent}, data: {data_ind}')

                    point_distance_sr[j, i] += loss_info['point_distance_sr']
                    point_distance_rs[j, i] += loss_info['point_distance_rs']
                    chamfer_loss_pcd[j, i] += loss_info['chamfer_loss_pcd']
                    particle_distance_sr[j, i] += loss_info['particle_distance_sr']
                    particle_distance_rs[j, i] += loss_info['particle_distance_rs']
                    chamfer_loss_particle[j, i] += loss_info['chamfer_loss_particle']
                    height_map_loss_pcd[j, i] += loss_info['height_map_loss_pcd']
                    emd_point_distance_loss[j, i] += loss_info['emd_point_distance_loss']
                    emd_particle_distance_loss[j, i] += loss_info['emd_particle_distance_loss']

            mpm_env.simulator.clear_ckpt()
            print(f'Time taken for data point {data_ind}: {time() - t0}')

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

loss_types = ['point_distance_sr', 'point_distance_rs', 'chamfer_loss_pcd',
              'particle_distance_sr', 'particle_distance_rs', 'chamfer_loss_particle',
              'height_map_loss_pcd',
              'emd_point_distance_loss', 'emd_particle_distance_loss'
              ]

for i in range(len(losses)):
    np.save(os.path.join(fig_data_path, f'{loss_types[i]}_{distance_type}_{xy_param}-{p_density_str}.npy'), losses[i])
    fig_title = (f'{loss_types[i]}\n'
                 f'yield_stress = {yield_stress}, rho = {rho}\n'
                 f'm_friction = 0.2, g_friction = 2.0')
    plot_loss_landscape(E, nu, losses[i], fig_title=fig_title,
                        loss_type=f'{loss_types[i]}_{distance_type}',
                        file_suffix=f'_{xy_param}-rightview-{p_density_str}',
                        view='right',
                        x_label='E', y_label='nu', z_label='Loss', show=False)
    plot_loss_landscape(E, nu, losses[i], fig_title=fig_title,
                        loss_type=f'{loss_types[i]}_{distance_type}',
                        file_suffix=f'_{xy_param}-leftview-{p_density_str}',
                        view='left',
                        x_label='E', y_label='nu', z_label='Loss', show=False)
    plot_loss_landscape(E, nu, losses[i], fig_title=fig_title,
                        loss_type=f'{loss_types[i]}_{distance_type}',
                        file_suffix=f'_{xy_param}-topview-{p_density_str}',
                        view='right',
                        x_label='E', y_label='nu', z_label='Loss', hm=True, show=False)
