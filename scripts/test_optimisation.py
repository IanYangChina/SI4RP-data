import os
import numpy as np
import open3d as o3d
import taichi as ti
ti.init(arch=ti.vulkan, device_memory_GB=5, default_fp=ti.f32, fast_math=False)
from doma.envs import SysIDEnv
from time import time

script_path = os.path.dirname(os.path.realpath(__file__))

horizon = int(0.03 / 0.002)
v = 0.045 / 0.03  # 1.5 m/s
trajectory_1 = np.zeros(shape=(horizon, 6))
trajectory_1[:5, 2] = -v
trajectory_1[5:, 2] = v

# Loading mesh
data_path = os.path.join(script_path, '..', 'data-motion-1', 'trial-1')
pcd_ind = str(0)
pcd_ind_ = str(1)
mesh_file_path = os.path.join(data_path, 'mesh_' + pcd_ind + '_repaired_normalised.obj')
centre_real = np.load(os.path.join(data_path, 'mesh_' + pcd_ind + '_repaired_centre.npy'))
centre_top_normalised = np.load(os.path.join(data_path, 'mesh_' + pcd_ind + '_repaired_normalised_centre_top.npy'))
centre_top_normalised_ = np.load(os.path.join(data_path, 'mesh_' + pcd_ind_ + '_repaired_normalised_centre_top.npy'))

# Building environment
initial_pos = (0.25, 0.25, centre_top_normalised[-1] + 0.01)
material_id = 2

env = SysIDEnv(ptcl_density=2e7, horizon=horizon,
               mesh_file=mesh_file_path, material_id=material_id, voxelise_res=1080, initial_pos=initial_pos,
               target_pcd_file=os.path.join(data_path, 'pcd_' + pcd_ind_ + '.ply'),
               pcd_offset=(-centre_real + initial_pos), mesh_offset=(0.25, 0.25, centre_top_normalised_[-1] + 0.01),
               target_mesh_file=os.path.join(data_path, 'mesh_' + pcd_ind_ + '_repaired_normalised.obj'),
               loss_weight=1.0, separate_param_grad=False)
mpm_env = env.mpm_env
# Update material parameters
# Initialising parameters
e = np.array([100])  # Young's modulus
nu = np.array([0.02])  # Poisson's ratio
yield_stress = np.array([4])
mpm_env.simulator.system_param_tmp[None].yield_stress = yield_stress
mpm_env.simulator.particle_param_tmp[material_id].E = e
mpm_env.simulator.particle_param_tmp[material_id].nu = nu
for s in range(horizon):
    mpm_env.simulator.system_param[s].yield_stress = yield_stress
    mpm_env.simulator.particle_param[s, material_id].E = e
    mpm_env.simulator.particle_param[s, material_id].nu = nu
env.reset()
print(mpm_env.simulator.n_particles_per_mat)
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
# mpm_env.apply_agent_action_p(np.array([0.25, 0.25, 2*centre_top_normalised[-1]+0.01, 0, 0, 45]))
for i in range(mpm_env.horizon):
    action = trajectory_1[i]
    mpm_env.step(action)
    env.render(mode='human')
    print(f'Step {i} chamfer loss: {mpm_env.loss.step_loss[i]}')

loss_info = mpm_env.get_final_loss()
t2 = time()

# backward
mpm_env.reset_grad()
mpm_env.get_final_loss_grad()
for i in range(horizon - 1, -1, -1):
    action = trajectory_1[i]
    mpm_env.step_grad(action=action)
# mpm_env.apply_agent_action_p_grad(np.array([0.25, 0.25, 2*centre_top_normalised[-1]+0.01, 0, 0, 45]))

t3 = time()
print(f'=======> forward: {t2 - t1:.2f}s backward: {t3 - t2:.2f}s')

# param_grad = mpm_env.simulator.get_param_grad()
# print(param_grad['particle_param_grad'][:, material_id, :])
print(mpm_env.loss.avg_point_distance_sr.grad)
print(mpm_env.loss.avg_point_distance_rs.grad)
print(mpm_env.simulator.system_param_tmp.grad[None].manipulator_friction)
print(mpm_env.simulator.system_param_tmp.grad[None].ground_friction)
print(mpm_env.simulator.system_param_tmp.grad[None].yield_stress)
print(mpm_env.simulator.particle_param_tmp.grad[material_id].E)
print(mpm_env.simulator.particle_param_tmp.grad[material_id].nu)
print(mpm_env.simulator.particle_param_tmp.grad[material_id].rho)
