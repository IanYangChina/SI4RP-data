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

script_path = os.path.dirname(os.path.realpath(__file__))
fig_data_path = os.path.join(script_path, '..', 'loss-landscapes')
DTYPE_NP = np.float32
DTYPE_TI = ti.f32
p_density = 3e7

ti.init(arch=ti.vulkan, device_memory_GB=6, default_fp=DTYPE_TI, fast_math=False, random_seed=1)
from doma.envs import SysIDEnv

def forward_backward(mpm_env, init_state, trajectory, backward=True,
                     render=False, render_init_pcd=False, render_end_pcd=False,
                     init_pcd_path=None, init_pcd_offset=None, init_mesh_path=None, init_mesh_pos=None):
    if render_init_pcd:
        x, _ = mpm_env.render(mode='point_cloud')
        RGBA = np.zeros((len(x), 4))
        RGBA[:, 0] = x[:, 2] / x[:, 2].max() * 255
        RGBA[:, 1] = x[:, 2] / x[:, 2].max() * 255
        RGBA[:, -1] = 255
        pts = Points(x, r=12, c=RGBA)

        real_pcd_pts = Points(init_pcd_path, r=12, c='r')
        x_ = real_pcd_pts.points() + init_pcd_offset
        x_[:, 0] += (x[:, 0].max() - x[:, 0].min() + 0.002)
        RGBA = np.zeros((len(x_), 4))
        RGBA[:, -1] = 255
        RGBA[:, 0] = x_[:, 2] / x_[:, 2].max() * 255
        RGBA[:, 1] = x_[:, 2] / x_[:, 2].max() * 255
        real_pcd_pts = Points(x_, r=12, c=RGBA)

        mesh = Mesh(init_mesh_path)
        coords = mesh.points()
        coords += init_mesh_pos
        coords[:, 0] += (x[:, 0].max() - x[:, 0].min() + 0.002 + x_[:, 0].max() - x_[:, 0].min() + 0.002)
        mesh.points(coords)

        show([pts, real_pcd_pts, mesh], __doc__, axes=True).close()    # Forward

    # plt.imshow(mpm_env.loss.height_map_pcd_target[None].to_numpy(), cmap='Greys')
    # plt.show()
    # plt.imshow(mpm_env.loss.height_map_particles_target[None].to_numpy(), cmap='Greys')
    # plt.show()

    t1 = time()
    mpm_env.set_state(init_state['state'], grad_enabled=True)
    for i in range(mpm_env.horizon):
        action = trajectory[i]
        mpm_env.step(action)
        # height_map = mpm_env.loss.height_maps[i].to_numpy()
        # plt.imshow(height_map, cmap='Greys')
        # plt.show()
        if render:
            mpm_env.render(mode='human')
            # img = mpm_env.render(mode='depth_array')
            # plt.imshow(img)
            # plt.show()
            sleep(0.1)
    loss_info = mpm_env.get_final_loss()
    for i, v in loss_info.items():
        if i != 'final_height_map':
            print(f'{i}: {v:.4f}')
        else:
            pass
            # plt.imshow(v.to_numpy(), cmap='Greys')
            # plt.show()

    t2 = time()

    if render_end_pcd:
        x, _ = mpm_env.render(mode='point_cloud')
        RGBA = np.zeros((len(x), 4))
        RGBA[:, -1] = 255
        RGBA[:, 0] = 250  # x[:, 0] / x[:, 0].max() * 255
        # RGBA[:, 0] = x[:, 1] / x[:, 1].max() * 255
        RGBA[:, 2] = x[:, 2] / x[:, 2].max() * 255
        pts = Points(x, r=12, c=RGBA)

        x_ = mpm_env.loss.target_pcd_points.to_numpy() / 1000
        x_[:, 0] += (x[:, 0].max() - x[:, 0].min() + 0.002)
        RGBA = np.zeros((mpm_env.loss.n_target_pcd_points, 4))
        RGBA[:, -1] = 255
        RGBA[:, 0] = 250  # x_[:, 0] / x_[:, 0].max() * 255
        # RGBA[:, 0] = x_[:, 1] / x_[:, 1].max() * 255
        RGBA[:, 2] = x_[:, 2] / x_[:, 2].max() * 255
        real_pcd_pts = Points(x_, r=12, c=RGBA)

        x__ = mpm_env.loss.target_particles_from_mesh.to_numpy() / 1000
        x__[:, 0] += (x[:, 0].max() - x[:, 0].min() + 0.002 + x_[:, 0].max() - x_[:, 0].min() + 0.002)
        RGBA = np.zeros((mpm_env.loss.n_target_particles_from_mesh, 4))
        RGBA[:, -1] = 255
        RGBA[:, 0] = 250  # x__[:, 0] / x__[:, 0].max() * 255
        # RGBA[:, 0] = x__[:, 1] / x__[:, 1].max() * 255
        RGBA[:, 2] = x__[:, 2] / x__[:, 2].max() * 255
        real_particle_pts = Points(x__, r=12, c=RGBA)

        mesh = Mesh(mpm_env.loss.target_mesh_path)
        coords = mesh.points()
        coords += mpm_env.loss.mesh_offset
        coords[:, 0] += (x[:, 0].max() - x[:, 0].min() + 0.002 + x_[:, 0].max() - x_[:, 0].min() + 0.002 + x__[:, 0].max() - x__[:, 0].min() + 0.002)
        mesh.points(coords)

        show([pts, real_pcd_pts, real_particle_pts, mesh], __doc__, axes=True).close()

    if backward:
        # backward
        mpm_env.reset_grad()
        mpm_env.get_final_loss_grad()
        for i in range(mpm_env.horizon - 1, -1, -1):
            action = trajectory[i]
            mpm_env.step_grad(action=action)

        t3 = time()
        print(f'=======> forward: {t2 - t1:.2f}s backward: {t3 - t2:.2f}s')


def make_env(data_path, data_ind, horizon, agent_name, material_id, cam_cfg):
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

    env = SysIDEnv(ptcl_density=p_density, horizon=horizon, material_id=material_id, voxelise_res=1080,
                   mesh_file=obj_start_mesh_file_path, initial_pos=obj_start_initial_pos,
                   target_pcd_file=obj_end_pcd_file_path,
                   pcd_offset=(-obj_start_centre_real + obj_start_initial_pos), down_sample_voxel_size=0.0035,
                   target_mesh_file=obj_end_mesh_file_path,
                   mesh_offset=(0.25, 0.25, obj_end_centre_top_normalised[-1] + 0.01),
                   loss_weight=1.0, separate_param_grad=False,
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


# Trajectory 2 presses down 0.02 m and lifts for 0.03 m
# In simulation we only takes the pressing down part
real_horizon_2 = int(0.04 / 0.001)
v = 0.05 / 0.04  # 1.25 m/s
horizon_down = int((0.02 / v) / 0.001)  # 8 steps
horizon_up = int((0.03 / v) / 0.001)  # 12 steps
horizon = horizon_down + horizon_up
trajectory = np.zeros(shape=(horizon, 6))
trajectory[:horizon_down, 2] = -v
trajectory[horizon_down:, 2] = v
agent = 'cylinder'
# Loading mesh
training_data_path = os.path.join(script_path, '..', 'data-motion-2', f'eef-{agent}')
data_ind = str(7)
material_id = 2
cam_cfg = {
    'pos': (0.25, -0.1, 0.2),
    'lookat': (0.25, 0.25, 0.05),
    'fov': 30,
    'lights': [{'pos': (0.5, -1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
               {'pos': (0.5, -1.5, 1.5), 'color': (0.5, 0.5, 0.5)}]
}
print(f'===> GPU memory before create env: {get_gpu_memory()}')
env, mpm_env, init_state = make_env(training_data_path, str(data_ind), horizon, agent, material_id, cam_cfg)
print(f'===> Num. simulation particles: {mpm_env.loss.n_particles_matching_mat}')
print(f'===> Num. target pcd points: {mpm_env.loss.n_target_pcd_points}')
print(f'===> Num. target particles: {mpm_env.loss.n_target_particles_from_mesh}')
print(f'===> GPU memory after create env: {get_gpu_memory()}')

E = np.array([40000], dtype=DTYPE_NP)
nu = np.array([0.45], dtype=DTYPE_NP)
yield_stress = np.array([1500], dtype=DTYPE_NP)

set_parameters(mpm_env, E, nu, yield_stress, rho=1000)

forward_backward(mpm_env, init_state, trajectory, backward=True, render=False,
                 render_init_pcd=False, render_end_pcd=False,
                 init_pcd_path=os.path.join(training_data_path, 'pcd_' + data_ind+str(0) + '.ply'),
                 init_pcd_offset=env.pcd_offset,
                 init_mesh_path=env.mesh_file,
                 init_mesh_pos=env.initial_pos)
print(f'GPU memory after forward-backward: {get_gpu_memory()}')

print(f"Gradient of E: {mpm_env.simulator.particle_param.grad[material_id].E}")
print(f"Gradient of nu: {mpm_env.simulator.particle_param.grad[material_id].nu}")
print(f"Gradient of rho: {mpm_env.simulator.particle_param.grad[material_id].rho}")
print(f"Gradient of yield stress: {mpm_env.simulator.system_param.grad[None].yield_stress}")
print(f"Gradient of manipulator friction: {mpm_env.simulator.system_param.grad[None].manipulator_friction}")
print(f"Gradient of ground friction: {mpm_env.simulator.system_param.grad[None].ground_friction}")
