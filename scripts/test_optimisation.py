import os
import numpy as np
import open3d as o3d
import taichi as ti
ti.init(arch=ti.vulkan, device_memory_GB=5, default_fp=ti.f32, fast_math=False)
from doma.envs import SysIDEnv
from time import time

script_path = os.path.dirname(os.path.realpath(__file__))

# Trajectory 1 presses down 0.015 m and lifts for 0.03 m
# In simulation we only takes the pressing down part
real_horizon_1 = int(0.03 / 0.001)
v = 0.045 / 0.03  # 1.5 m/s
horizon_1 = int((0.015 / v) / 0.001)  # 5 steps
trajectory_1 = np.zeros(shape=(horizon_1, 6))
trajectory_1[:, 2] = -v
agent_1 = 'rectangle'

# Trajectory 2 presses down 0.02 m and lifts for 0.03 m
# In simulation we only takes the pressing down part
real_horizon_2 = int(0.04 / 0.001)
v = 0.05 / 0.04  # 1.25 m/s
horizon_2 = int((0.02 / v) / 0.001)  # 8 steps
trajectory_2 = np.zeros(shape=(horizon_2, 6))
trajectory_2[:, 2] = -v
agent_2 = 'round'

agent = agent_1
horizon = horizon_1
trajectory = trajectory_1
# Loading mesh
data_path = os.path.join(script_path, '..', 'data-motion-1', 'eef-1')
data_ind = str(0)
mesh_file_path = os.path.join(data_path, 'mesh_' + data_ind+str(0) + '_repaired_normalised.obj')
centre_real = np.load(os.path.join(data_path, 'mesh_' + data_ind+str(0) + '_repaired_centre.npy'))
centre_top_normalised = np.load(os.path.join(data_path, 'mesh_' + data_ind+str(0) + '_repaired_normalised_centre_top.npy'))
centre_top_normalised_ = np.load(os.path.join(data_path, 'mesh_' + data_ind+str(1) + '_repaired_normalised_centre_top.npy'))

# Building environment
initial_pos = (0.25, 0.25, centre_top_normalised[-1] + 0.01)
agent_init_pos = (0.25, 0.25, 2*centre_top_normalised[-1]+0.01)
material_id = 2

env = SysIDEnv(ptcl_density=1e7, horizon=horizon,
               mesh_file=mesh_file_path, material_id=material_id, voxelise_res=1080, initial_pos=initial_pos,
               target_pcd_file=os.path.join(data_path, 'pcd_' + data_ind+str(1) + '.ply'),
               pcd_offset=(-centre_real + initial_pos), mesh_offset=(0.25, 0.25, centre_top_normalised_[-1] + 0.01),
               target_mesh_file=os.path.join(data_path, 'mesh_' + data_ind+str(1) + '_repaired_normalised.obj'),
               loss_weight=1.0, separate_param_grad=False,
               agent_cfg_file=agent+'_eef.yaml', agent_init_pos=agent_init_pos, agent_init_euler=(0, 0, 0))
mpm_env = env.mpm_env
# Update material parameters
# Initialising parameters
e = np.array([100], dtype=np.float32)  # Young's modulus
nu = np.array([0.02], dtype=np.float32)  # Poisson's ratio
yield_stress = np.array([4], dtype=np.float32)
mpm_env.simulator.system_param[None].yield_stress = yield_stress
mpm_env.simulator.particle_param[material_id].E = e
mpm_env.simulator.particle_param[material_id].nu = nu
env.reset()
print(mpm_env.loss.n_target_pcd_points)
print(mpm_env.loss.n_target_particles_from_mesh)
init_state = mpm_env.get_state()

# frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
# pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mpm_env.loss.target_pcd_points_np)).paint_uniform_color([1, 0, 0])
# pcd_ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mpm_env.loss.target_particles_from_mesh_np)).paint_uniform_color([0, 1, 0])
# o3d.visualization.draw_geometries([pcd, pcd_, frame])
# exit()

# Forward
t1 = time()
mpm_env.set_state(init_state['state'], grad_enabled=True)
for i in range(mpm_env.horizon):
    action = trajectory[i]
    mpm_env.step(action)
    env.render(mode='human')
    print(f'Step {i} chamfer loss: {mpm_env.loss.step_loss[i]}')

loss_info = mpm_env.get_final_loss()
t2 = time()

# backward
mpm_env.reset_grad()
mpm_env.get_final_loss_grad()
for i in range(horizon - 1, -1, -1):
    action = trajectory[i]
    mpm_env.step_grad(action=action)

t3 = time()
print(f'=======> forward: {t2 - t1:.2f}s backward: {t3 - t2:.2f}s')

param_grad = mpm_env.simulator.get_param_grad()
print(param_grad['particle_param_grad'][material_id, :])
print(param_grad['system_param_grad'])
print(mpm_env.loss.avg_point_distance_sr.grad)
print(mpm_env.loss.avg_point_distance_rs.grad)
