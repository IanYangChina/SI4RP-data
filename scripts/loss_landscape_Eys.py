import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import os
import taichi as ti
from doma.envs import SysIDEnv
from time import time
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 12})
script_path = os.path.dirname(os.path.realpath(__file__))
fig_data_path = os.path.join(script_path, '..', 'loss-landscapes')
DTYPE_NP = np.float32
DTYPE_TI = ti.f32


def plot_loss_landscape(p1, p2, loss, fig_title='Fig', loss_type='bi_chamfer_loss', file_suffix='', view='left',
                        x_label='p1', y_label='p2', z_label='Loss', show=False):
    if not show:
        mpl.use('Agg')
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Plot the surface.
    surf = ax.plot_surface(p1, p2, loss, cmap=cm.gist_earth,
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

    plt.savefig(os.path.join(fig_data_path, f"{loss_type}_landscape{file_suffix}.pdf"), dpi=500, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

# for i in range(len(loss_types)):
#     loss = np.load(os.path.join(fig_data_path, f'{loss_types[i]}_{distance_type}_{xy_param}-{p_density_str}.npy'))
#     fig_title = f'{loss_types[i]} with yield_stress = {yield_stress}'
#     plot_loss_landscape(E, nu, loss, fig_title=fig_title,
#                         loss_type=f'{loss_types[i]}_{distance_type}',
#                         file_suffix=f'_{xy_param}-rightview-{p_density_str}',
#                         view='right',
#                         x_label='E', y_label='nu', z_label='Loss', show=False)
#     plot_loss_landscape(E, nu, loss, fig_title=fig_title,
#                         loss_type=f'{loss_types[i]}_{distance_type}',
#                         file_suffix=f'_{xy_param}-leftview-{p_density_str}',
#                         view='left',
#                         x_label='E', y_label='nu', z_label='Loss', show=False)
# exit()


def make_env(data_path, data_ind, horizon, agent_name, loss_config, cam_cfg=None):
    obj_start_mesh_file_path = os.path.join(data_path, 'mesh_' + data_ind+str(0) + '_repaired_normalised.obj')
    if not os.path.exists(obj_start_mesh_file_path):
        return None, None
    obj_start_centre_real = np.load(os.path.join(data_path, 'mesh_' + data_ind+str(0) + '_repaired_centre.npy')).astype(DTYPE_NP)
    obj_start_centre_top_normalised = np.load(
        os.path.join(data_path, 'mesh_' + data_ind+str(0) + '_repaired_normalised_centre_top.npy')).astype(DTYPE_NP)

    obj_end_pcd_file_path = os.path.join(data_path, 'pcd_' + data_ind+str(1) + '.ply')
    obj_end_mesh_file_path = os.path.join(data_path, 'mesh_' + data_ind+str(1) + '_repaired_normalised.obj')
    obj_end_centre_top_normalised = np.load(
        os.path.join(data_path, 'mesh_' + data_ind+str(1) + '_repaired_normalised_centre_top.npy')).astype(DTYPE_NP)

    # Building environment
    obj_start_initial_pos = np.array([0.25, 0.25, obj_start_centre_top_normalised[-1] + 0.01], dtype=DTYPE_NP)
    agent_init_pos = (0.25, 0.25, 2*obj_start_centre_top_normalised[-1] + 0.01)
    height_map_res = loss_config['height_map_res']
    loss_config.update({
        'target_pcd_path': obj_end_pcd_file_path,
        'pcd_offset': (-obj_start_centre_real + obj_start_initial_pos),
        'target_mesh_file': obj_end_mesh_file_path,
        'mesh_offset': (0.25, 0.25, obj_end_centre_top_normalised[-1] + 0.01),
        'target_pcd_height_map_path': os.path.join(data_path,
                                                   f'target_pcd_height_map-{data_ind}-res{str(height_map_res)}-vdsize{str(0.001)}.npy'),
    })

    env = SysIDEnv(ptcl_density=p_density, horizon=horizon, material_id=2, voxelise_res=1080,
                   mesh_file=obj_start_mesh_file_path, initial_pos=obj_start_initial_pos,
                   loss_cfg=loss_config,
                   agent_cfg_file=agent_name+'_eef.yaml', agent_init_pos=agent_init_pos, agent_init_euler=(0, 0, 0),
                   render_agent=True, camera_cfg=cam_cfg)
    env.reset()
    mpm_env = env.mpm_env
    init_state = mpm_env.get_state()

    return env, mpm_env, init_state


def set_parameters(mpm_env, E, nu, yield_stress):
    mpm_env.simulator.system_param[None].yield_stress = yield_stress.copy()
    mpm_env.simulator.particle_param[2].E = E.copy()
    mpm_env.simulator.particle_param[2].nu = nu.copy()
    mpm_env.simulator.particle_param[2].rho = 1000


p_density = 1e7
p_density_str = '1e7pd'

loss_cfg = {
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
    'emd_point_distance_rs_loss': True,
}

# Trajectory 1 presses down 0.015 m and lifts for 0.03 m
# Trajectory 2 presses down 0.02 m and lifts for 0.03 m

# Setting up horizon and trajectory
dt = 0.001
# Trajectory 1 press down 0.015 m and lifts for 0.03 m
# In simulation we only simulate the pressing down part
real_horizon_1 = int(0.03 / dt)
v = 0.045 / 0.03  # 1.5 m/s
horizon_1_up = int((0.015 / v) / dt)  # 0.01 s
horizon_1_down = int((0.03 / v) / dt)  # 0.02 s
horizon_1 = horizon_1_up + horizon_1_down  # 30 steps
trajectory_1 = np.zeros(shape=(horizon_1, 6), dtype=DTYPE_NP)
trajectory_1[:horizon_1_up, 2] = -v
trajectory_1[horizon_1_up:, 2] = v

# Trajectory 2 press down 0.02 m and lifts for 0.03 m
# In simulation we only simulate the pressing down part
real_horizon_2 = int(0.04 / dt)
v = 0.05 / 0.045  # 1.11111111 m/s
horizon_2_up = int((0.02 / v) / dt)  # 0.018 s
horizon_2_down = int((0.03 / v) / dt)  # 0.027 s
horizon_2 = horizon_2_up + horizon_2_down  # 45 steps
trajectory_2 = np.zeros(shape=(horizon_2, 6), dtype=DTYPE_NP)
trajectory_2[:horizon_2_up, 2] = -v
trajectory_2[horizon_2_up:, 2] = v

xy_param = 'E-yieldstress'
E_list = np.arange(10000, 100000, 2000).astype(DTYPE_NP)
yield_stress_list = np.arange(50, 2050, 40).astype(DTYPE_NP)

E, yield_stress = np.meshgrid(E_list, yield_stress_list)
nu = np.array([0.45], dtype=DTYPE_NP)

avg_point_distance_sr = np.zeros_like(E)
avg_point_distance_rs = np.zeros_like(E)
chamfer_loss_pcd = np.zeros_like(E)
avg_particle_distance_sr = np.zeros_like(E)
avg_particle_distance_rs = np.zeros_like(E)
chamfer_loss_particle = np.zeros_like(E)
height_map_loss_pcd = np.zeros_like(E)
emd_loss = np.zeros_like(E)

n_datapoints = 9
for agent in ['rectangle', 'round', 'cylinder']:
    training_data_path = os.path.join(script_path, '..', 'data-motion-1', f'eef-{agent}')
    for data_ind in range(n_datapoints):
        ti.reset()
        ti.init(arch=ti.opengl, default_ip=ti.i32, default_fp=DTYPE_TI, fast_math=False, random_seed=1)
        env, mpm_env, init_state = make_env(training_data_path, str(data_ind), horizon_1, agent, loss_cfg)
        print(f'===> Num. simulation particles: {mpm_env.loss.n_particles_matching_mat}')
        print(f'===> Num. target pcd points: {mpm_env.loss.n_target_pcd_points}')
        print(f'===> Num. target particles: {mpm_env.loss.n_target_particles_from_mesh}')
        t0 = time()
        print(f'Start calculating losses with grid size: {avg_point_distance_sr.shape}')
        for i in range(len(E_list)):
            for j in range(len(yield_stress_list)):
                set_parameters(mpm_env, E_list[i], nu, yield_stress_list[j])
                mpm_env.set_state(init_state['state'], grad_enabled=False)
                for k in range(mpm_env.horizon):
                    action = trajectory_1[k]
                    mpm_env.step(action)
                loss_info = mpm_env.get_final_loss()

                print(f'The {i}, {j}-th loss is:')
                for b, v in loss_info.items():
                    if b == 'final_height_map':
                        continue
                    print(f'{b}: {v:.4f}')
                avg_point_distance_sr[j, i] += loss_info['avg_point_distance_sr']
                avg_point_distance_rs[j, i] += loss_info['avg_point_distance_rs']
                chamfer_loss_pcd[j, i] += loss_info['chamfer_loss_pcd']
                avg_particle_distance_sr[j, i] += loss_info['avg_particle_distance_sr']
                avg_particle_distance_rs[j, i] += loss_info['avg_particle_distance_rs']
                chamfer_loss_particle[j, i] += loss_info['chamfer_loss_particle']
                height_map_loss_pcd[j, i] += loss_info['height_map_loss_pcd']
                emd_loss[j, i] += loss_info['emd_loss']
                
        mpm_env.simulator.clear_ckpt()
        print(f'Time taken for data point {data_ind}: {time() - t0}')

distance_type = 'exponential'
losses = [avg_point_distance_sr / (3 * n_datapoints),
          avg_point_distance_rs / (3 * n_datapoints),
          chamfer_loss_pcd / (3 * n_datapoints),
          avg_particle_distance_sr / (3 * n_datapoints),
          avg_particle_distance_rs / (3 * n_datapoints),
          chamfer_loss_particle / (3 * n_datapoints),
          height_map_loss_pcd / (3 * n_datapoints),
          emd_loss / (3 * n_datapoints)]

loss_types = ['avg_point_distance_sr', 'avg_point_distance_rs', 'chamfer_loss_pcd',
              'avg_particle_distance_sr', 'avg_particle_distance_rs', 'chamfer_loss_particle',
              'height_map_loss_pcd', 'emd_loss']

for i in range(len(losses)):
    np.save(os.path.join(fig_data_path, f'{loss_types[i]}_{distance_type}_{xy_param}-{p_density_str}.npy'), losses[i])
    fig_title = f'{loss_types[i]} with nu = {nu}'
    plot_loss_landscape(E, yield_stress, losses[i], fig_title=fig_title,
                        loss_type=f'{loss_types[i]}_{distance_type}',
                        file_suffix=f'_{xy_param}-rightview-{p_density_str}',
                        view='right',
                        x_label='E', y_label='yield_stress', z_label='Loss', show=False)
    plot_loss_landscape(E, yield_stress, losses[i], fig_title=fig_title,
                        loss_type=f'{loss_types[i]}_{distance_type}',
                        file_suffix=f'_{xy_param}-leftview-{p_density_str}',
                        view='left',
                        x_label='E', y_label='yield_stress', z_label='Loss', show=False)
