import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import os
import taichi as ti
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


E_list = np.arange(10000, 100000, 2000).astype(DTYPE_NP)
nu_list = np.arange(0.01, 0.48, 0.01).astype(DTYPE_NP)
yield_stress_list = np.arange(50, 2050, 40).astype(DTYPE_NP)

E, nu = np.meshgrid(E_list, nu_list)
# E = np.array([30000], dtype=DTYPE_NP)
# nu = np.array([0.45], dtype=DTYPE_NP)
yield_stress = np.array([1500.0], dtype=DTYPE_NP)
xy_param = 'E-nu'
p_density = 3e7
p_density_str = '3e7pd'

distance_type = 'exponential'
loss_types = ['d_pcd_sr_loss', 'd_pcd_rs_loss', 'd_pcd_total', 'd_particle_sr_loss', 'd_particle_rs_loss', 'd_particle_total']

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

ti.init(arch=ti.vulkan, device_memory_GB=8, default_fp=DTYPE_TI, fast_math=False, random_seed=1)
from doma.envs import SysIDEnv


def make_env(data_path, data_ind, horizon, agent_name):
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

    env = SysIDEnv(ptcl_density=p_density, horizon=horizon, material_id=2, voxelise_res=1080,
                   mesh_file=obj_start_mesh_file_path, initial_pos=obj_start_initial_pos,
                   target_pcd_file=obj_end_pcd_file_path,
                   pcd_offset=(-obj_start_centre_real + obj_start_initial_pos),
                   target_mesh_file=obj_end_mesh_file_path,
                   mesh_offset=(0.25, 0.25, obj_end_centre_top_normalised[-1] + 0.01),
                   loss_weight=1.0, separate_param_grad=False,
                   agent_cfg_file=agent_name+'_eef.yaml', agent_init_pos=agent_init_pos, agent_init_euler=(0, 0, 45))
    env.reset()
    mpm_env = env.mpm_env
    init_state = mpm_env.get_state()

    return mpm_env, init_state


def set_parameters(mpm_env, E, nu, yield_stress):
    mpm_env.simulator.system_param[None].yield_stress = yield_stress.copy()
    mpm_env.simulator.particle_param[2].E = E.copy()
    mpm_env.simulator.particle_param[2].nu = nu.copy()
    mpm_env.simulator.particle_param[2].rho = 1000


# Trajectory 1 presses down 0.015 m and lifts for 0.03 m
# Trajectory 2 presses down 0.02 m and lifts for 0.03 m

real_horizon = int(0.04 / 0.001)
v = 0.05 / 0.04  # 1.5 m/s
horizon_down = int((0.02 / v) / 0.001)  # 5 steps
horizon_up = int((0.03 / v) / 0.001)  # 5 steps
horizon = horizon_down + horizon_up
trajectory = np.zeros(shape=(horizon, 6))
trajectory[:horizon_down, 2] = -v
trajectory[horizon_down:, 2] = v
agent = 'round'

# Loading mesh
training_data_path = os.path.join(script_path, '..', 'data-motion-2', f'eef-{agent}')
data_ind = str(5)
material_id = 2
mpm_env, init_state = make_env(training_data_path, str(data_ind), horizon, agent)
d_pcd_sr_loss = np.zeros_like(E)
d_pcd_rs_loss = np.zeros_like(E)
d_pcd_total = np.zeros_like(E)
d_particle_sr_loss = np.zeros_like(E)
d_particle_rs_loss = np.zeros_like(E)
d_particle_total = np.zeros_like(E)

t0 = time()
print(f'Start calculating losses with grid size: {d_pcd_sr_loss.shape}')
for i in range(len(E_list)):
    for j in range(len(nu_list)):
        set_parameters(mpm_env, E_list[i], nu_list[j], yield_stress)
        mpm_env.set_state(init_state['state'], grad_enabled=False)
        for k in range(mpm_env.horizon):
            action = trajectory[k]
            mpm_env.step(action)
        loss_info = mpm_env.get_final_loss()
        print(f'The {i}, {j}-th loss is:')
        for b, v in loss_info.items():
            print(f'{b}: {v:.4f}')
        d_pcd_sr_loss[j, i] = loss_info['avg_point_distance_sr']
        d_pcd_rs_loss[j, i] = loss_info['avg_point_distance_rs']
        d_pcd_total[j, i] = loss_info['chamfer_loss_pcd']
        d_particle_sr_loss[j, i] = loss_info['avg_particle_distance_sr']
        d_particle_rs_loss[j, i] = loss_info['avg_particle_distance_rs']
        d_particle_total[j, i] = loss_info['chamfer_loss_particle']

print(f'Time taken: {time() - t0}')

distance_type = 'exponential'
losses = [d_pcd_sr_loss, d_pcd_rs_loss, d_pcd_total, d_particle_sr_loss, d_particle_rs_loss, d_particle_total]
loss_types = ['d_pcd_sr_loss', 'd_pcd_rs_loss', 'd_pcd_total', 'd_particle_sr_loss', 'd_particle_rs_loss', 'd_particle_total']

for i in range(len(losses)):
    np.save(os.path.join(fig_data_path, f'{loss_types[i]}_{distance_type}_{xy_param}-{p_density_str}.npy'), losses[i])
    fig_title = f'{loss_types[i]} with yield_stress = {yield_stress}'
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
