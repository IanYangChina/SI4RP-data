import os
import numpy as np
import open3d as o3d
from doma.envs import ClayEnv
from time import time

script_path = os.path.dirname(os.path.realpath(__file__))

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

# Initialising parameters
mu = np.array([416.6])
lamda = np.array([277.78])
yield_stress = np.array([20])

horizon = int(0.03 / 0.002)
v = 0.045 / 0.03  # 1.5 m/s
trajectory_1 = np.zeros(shape=(horizon, 6))
trajectory_1[:5, 2] = -v
trajectory_1[5:, 2] = v

# Loading mesh
data_path = os.path.join(script_path, '..', 'data-motion-1', 'trial-1')
pcd_ind = str(1)
pcd_ind_ = str(2)
mesh_file_path = os.path.join(data_path, 'mesh_'+pcd_ind+'_repaired_normalised.obj')
centre_real = np.load(os.path.join(data_path, 'mesh_'+pcd_ind+'_repaired_centre.npy'))
centre_top_normalised = np.load(os.path.join(data_path, 'mesh_'+pcd_ind+'_repaired_normalised_centre_top.npy'))

# Building environment
initial_pos = (0.25, 0.25, centre_top_normalised[-1]+0.01)
material_id = 2

env = ClayEnv(ptcl_density=1e7, horizon=horizon,
              mesh_file=mesh_file_path, material_id=material_id, voxelise_res=1080, initial_pos=initial_pos,
              target_pcd_file=os.path.join(data_path, 'pcd_'+pcd_ind_+'.ply'),
              pcd_offset=(-centre_real+initial_pos))
env.reset()
mpm_env = env.mpm_env
print(mpm_env.simulator.n_particles_per_mat)
print(mpm_env.loss.n_target_pcd_points)
init_state = mpm_env.get_state()
# Update material parameters
mpm_env.simulator.system_param[None].yield_stress = yield_stress
mpm_env.simulator.particle_param[material_id].mu = mu
mpm_env.simulator.particle_param[material_id].lamda = lamda

# Forward
t1 = time()
mpm_env.set_state(init_state['state'], grad_enabled=True)
mpm_env.apply_agent_action_p(np.array([0.25, 0.25, 2*centre_top_normalised[-1]+0.01, 0, 0, 45]))
for i in range(horizon):
    action = trajectory_1[i]
    mpm_env.step(action)
    env.render(mode='human')
    print(f'Step {i} chamfer loss: {mpm_env.loss.step_loss[i]}')

loss_info = mpm_env.get_final_loss()
t2 = time()

print(f'Forward: {t2 - t1:.2f}s, loss: {loss_info}')

# backward
mpm_env.reset_grad()
mpm_env.get_final_loss_grad()
for i in range(horizon-1, -1, -1):
    action = trajectory_1[i]
    mpm_env.step_grad(action=action)
mpm_env.apply_agent_action_p_grad(np.array([0.25, 0.25, 2*centre_top_normalised[-1]+0.01, 0, 0, 45]))

param_grad = mpm_env.simulator.get_param_grad()

t3 = time()
print(f'=======> forward: {t2 - t1:.2f}s backward: {t3 - t2:.2f}s')
print(param_grad)
print(mpm_env.loss.total_loss.grad)
print(mpm_env.loss.step_loss.grad)
print(mpm_env.loss.chamfer_loss.grad)
