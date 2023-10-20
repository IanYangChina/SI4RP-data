import numpy as np
import os
import taichi as ti
from time import time

script_path = os.path.dirname(os.path.realpath(__file__))
fig_data_path = os.path.join(script_path, '..', 'loss-landscapes')
DTYPE_NP = np.float32
DTYPE_TI = ti.f32
p_density = 2e7

ti.init(arch=ti.vulkan, device_memory_GB=8, default_fp=DTYPE_TI, fast_math=False, random_seed=1)
from doma.envs import SysIDEnv


def forward_backward(mpm_env, init_state, trajectory, render, backward=True):
    # Forward
    t1 = time()
    mpm_env.set_state(init_state['state'], grad_enabled=True)
    for i in range(mpm_env.horizon):
        action = trajectory[i]
        mpm_env.step(action)
        if render:
            mpm_env.render(mode='human')
    loss_info = mpm_env.get_final_loss()
    for i, v in loss_info.items():
        print(f'{i}: {v:.4f}')
    t2 = time()

    if backward:
        # backward
        mpm_env.reset_grad()
        mpm_env.get_final_loss_grad()
        for i in range(mpm_env.horizon - 1, -1, -1):
            action = trajectory[i]
            mpm_env.step_grad(action=action)

        t3 = time()
        print(f'=======> forward: {t2 - t1:.2f}s backward: {t3 - t2:.2f}s')


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


# Trajectory 1 presses down 0.015 m and lifts for 0.03 m
# In simulation we only takes the pressing down part
real_horizon_1 = int(0.03 / 0.001)
v = 0.045 / 0.03  # 1.5 m/s
horizon_1 = int((0.015 / v) / 0.001)  # 5 steps
trajectory_1 = np.zeros(shape=(horizon_1, 6))
trajectory_1[:, 2] = -v
agent_1 = 'rectangle'

agent = agent_1
horizon = horizon_1
trajectory = trajectory_1
# Loading mesh
training_data_path = os.path.join(script_path, '..', 'data-motion-1', f'eef-{agent}')
data_ind = str(0)
material_id = 2
mpm_env, init_state = make_env(training_data_path, str(data_ind), horizon, agent)

E = np.array([150], dtype=DTYPE_NP)
nu = np.array([0.2], dtype=DTYPE_NP)
yield_stress = np.array([1.0], dtype=DTYPE_NP)
set_parameters(mpm_env, E, nu, yield_stress)

forward_backward(mpm_env, init_state, trajectory, render=False, backward=True)
print(f"Gradient of E: {mpm_env.simulator.particle_param.grad[material_id].E}")
print(f"Gradient of nu: {mpm_env.simulator.particle_param.grad[material_id].nu}")
print(f"Gradient of rho: {mpm_env.simulator.particle_param.grad[material_id].rho}")
print(f"Gradient of yield stress: {mpm_env.simulator.system_param.grad[None].yield_stress}")
print(f"Gradient of manipulator friction: {mpm_env.simulator.system_param.grad[None].manipulator_friction}")
print(f"Gradient of ground friction: {mpm_env.simulator.system_param.grad[None].ground_friction}")

