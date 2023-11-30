import numpy as np
import os
import taichi as ti
from time import time, sleep
import open3d as o3d
from vedo import Points, show, Mesh
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pylab as plt
from doma.engine.utils.misc import get_gpu_memory
import psutil
process = psutil.Process(os.getpid())

script_path = os.path.dirname(os.path.realpath(__file__))
DTYPE_NP = np.float32
DTYPE_TI = ti.f32
p_density = 1e7
loss_cfg = {
    'point_distance_rs_loss': True,
    'point_distance_sr_loss': False,
    'down_sample_voxel_size': 0.004,
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

from doma.envs import SysIDEnv


def make_env(data_path, data_ind, horizon, agent_name, cam_cfg, loss_config):
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
    agent_init_pos = (0.20, 0.25, obj_start_centre_top_normalised[-1] + 0.01)
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


def set_parameters(mpm_env, E, nu, yield_stress, rho=None):
    mpm_env.simulator.system_param[None].yield_stress = yield_stress.copy()
    mpm_env.simulator.particle_param[2].E = E.copy()
    mpm_env.simulator.particle_param[2].nu = nu.copy()
    if rho is not None:
        mpm_env.simulator.particle_param[2].rho = rho


# Moves the gripper at +x direction for 0.1 s
# Increase speed to 1.0 m/s in the first 0.01 s
# Decrease speed to 0.0 m/s in the last 0.01 s
horizon = int(0.1 / 0.001)
v = 1.0  # m/s
trajectory = np.zeros(shape=(horizon, 6))
acceleration = np.zeros(shape=(horizon, 6))
for i in range(horizon):
    if i < 10:
        trajectory[i, 0] = v * i / 10
    elif i >= horizon - 10:
        trajectory[i, 0] = v * (horizon - i - 1) / 10
    else:
        trajectory[i, 0] = v
for i in range(horizon-1):
    acceleration[i, 0] = (trajectory[i+1, 0] - trajectory[i, 0]) / 0.001

print(trajectory)
print(acceleration)
exit()

agent = 'rectangle'
training_data_path = os.path.join(script_path, '..', 'data-motion-2', f'eef-{agent}')
data_ind = 2
cam_cfg = {
    'pos': (0.25, -0.2, 0.2),
    'lookat': (0.25, 0.25, 0.05),
    'fov': 30,
    'lights': [{'pos': (0.5, -1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
               {'pos': (0.5, -1.5, 1.5), 'color': (0.5, 0.5, 0.5)}]
}

ti.reset()
ti.init(arch=ti.opengl, default_fp=DTYPE_TI, default_ip=ti.i32, fast_math=False, random_seed=1)
print(f'===> CPU memory occupied before create env: {process.memory_percent()} %')
print(f'===> GPU memory before create env: {get_gpu_memory()}')
env, mpm_env, init_state = make_env(training_data_path, str(data_ind), horizon, agent, cam_cfg, loss_cfg.copy())
print(f'===> Num. simulation particles: {mpm_env.loss.n_particles_matching_mat}')
print(f'===> Num. target pcd points: {mpm_env.loss.n_target_pcd_points}')
print(f'===> Num. target particles: {mpm_env.loss.n_target_particles_from_mesh}')
print(f'===> CPU memory occupied after create env: {process.memory_percent()} %')
print(f'===> GPU memory after create env: {get_gpu_memory()}')

E = np.array([40000], dtype=DTYPE_NP)
nu = np.array([0.45], dtype=DTYPE_NP)
yield_stress = np.array([1500], dtype=DTYPE_NP)

set_parameters(mpm_env, E, nu, yield_stress, rho=1000)

render = True
mpm_env.set_state(init_state['state'], grad_enabled=True)
for i in range(mpm_env.horizon):
    action = trajectory[i]
    mpm_env.step(action)
    if render:
        mpm_env.render(mode='human')
        sleep(0.05)
loss_info = mpm_env.get_final_loss()
for i, v in loss_info.items():
    if i != 'final_height_map':
        print(f'===> {i}: {v:.4f}')
    else:
        pass