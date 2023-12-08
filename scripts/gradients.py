import numpy as np
import os
import taichi as ti
from time import time, sleep
import open3d as o3d
from vedo import Points, show, Mesh
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from doma.engine.utils.misc import get_gpu_memory
import psutil
import json

process = psutil.Process(os.getpid())
script_path = os.path.dirname(os.path.realpath(__file__))
gradient_file_path = os.path.join(script_path, '..', 'gradients')
os.makedirs(gradient_file_path, exist_ok=True)

DTYPE_NP = np.float32
DTYPE_TI = ti.f32
p_density = 6e7
loss_cfg = {
    'exponential_distance': False,
    'point_distance_rs_loss': False,
    'point_distance_sr_loss': False,
    'down_sample_voxel_size': 0.005,
    'particle_distance_rs_loss': False,
    'particle_distance_sr_loss': False,
    'voxelise_res': 1080,
    'ptcl_density': p_density,
    'load_height_map': True,
    'height_map_loss': False,
    'height_map_res': 32,
    'height_map_size': 0.11,
    'emd_point_distance_rs_loss': False,
}

from doma.envs import SysIDEnv

def forward_backward(mpm_env, init_state, trajectory, backward=True,
                     render=False, render_init_pcd=False, render_end_pcd=False, render_heightmap=False,
                     init_pcd_path=None, init_pcd_offset=None, init_mesh_path=None, init_mesh_pos=None):
    cmap = 'Greys'
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

    t1 = time()
    mpm_env.set_state(init_state['state'], grad_enabled=True)
    for i in range(mpm_env.horizon):
        action = trajectory[i]
        mpm_env.step(action)
        if render:
            mpm_env.render(mode='human')
            # img = mpm_env.render(mode='depth_array')
            # plt.imshow(img)
            # plt.show()
            # sleep(0.1)
    loss_info = mpm_env.get_final_loss()
    for i, v in loss_info.items():
        if i != 'final_height_map':
            print(f'===> {i}: {v:.4f}')
        else:
            pass
    if render_heightmap:
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_title('Target PCD\nheight map')
        im1 = ax1.imshow(mpm_env.loss.height_map_pcd_target.to_numpy(), cmap=cmap)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='2%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.set_title('Simulated particle\nheight map')
        im2 = ax2.imshow(loss_info['final_height_map'], cmap=cmap)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='2%', pad=0.05)
        fig.colorbar(im2, cax=cax, orientation='vertical')

        plt.show()
        plt.close()
        del fig, ax1, ax2, im1, im2, cax, divider

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
        print(f'===> forward: {t2 - t1:.2f}s backward: {t3 - t2:.2f}s')


def make_env(data_path, data_ind, horizon, dt_global, agent_name, material_id, cam_cfg, loss_config):
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

    env = SysIDEnv(ptcl_density=p_density, horizon=horizon, dt_global=dt_global, material_id=material_id, voxelise_res=1080,
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
    mpm_env.simulator.particle_param[2].rho = 1300


# Trajectory 2 presses down 0.02 m and lifts for 0.03 m

trajectory = np.load(os.path.join(script_path, '..', 'data-motion-1', 'eef_v_trajectory_.npy'))
horizon = 150
dt_global = 1.03 / trajectory.shape[0]

agent = 'cylinder'
# Loading mesh
training_data_path = os.path.join(script_path, '..', 'data-motion-1', f'eef-{agent}')
material_id = 2
cam_cfg = {
    'pos': (0.25, -0.1, 0.2),
    'lookat': (0.25, 0.25, 0.05),
    'fov': 30,
    'lights': [{'pos': (0.5, -1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
               {'pos': (0.5, -1.5, 1.5), 'color': (0.5, 0.5, 0.5)}]
}

E_range = (10000, 100000)
nu_range = (0.001, 0.49)
yield_stress_range = (50, 3000)

avg_grads = np.zeros(shape=(5,), dtype=DTYPE_NP)

for data_ind in range(9):
    ti.reset()
    ti.init(arch=ti.opengl,
            # device_memory_GB=2, ad_stack_size=1024,
            # offline_cache=True, log_level=ti.TRACE,
            default_fp=DTYPE_TI, default_ip=ti.i32,
            fast_math=False, random_seed=1,
            # debug=True, check_out_of_bound=True
            )
    print(f'===> CPU memory occupied before create env: {process.memory_percent()} %')
    print(f'===> GPU memory before create env: {get_gpu_memory()}')
    env, mpm_env, init_state = make_env(training_data_path, str(data_ind), horizon, dt_global, agent, material_id, cam_cfg, loss_cfg.copy())
    print(f'===> Num. simulation particles: {mpm_env.loss.n_particles_matching_mat}')
    print(f'===> Num. target pcd points: {mpm_env.loss.n_target_pcd_points}')
    print(f'===> Num. target particles: {mpm_env.loss.n_target_particles_from_mesh}')
    print(f'===> CPU memory occupied after create env: {process.memory_percent()} %')
    print(f'===> GPU memory after create env: {get_gpu_memory()}')

    for _ in range(5):
        E = np.asarray(np.random.uniform(E_range[0], E_range[1]), dtype=DTYPE_NP).reshape((1,))  # Young's modulus
        nu = np.asarray(np.random.uniform(nu_range[0], nu_range[1]), dtype=DTYPE_NP).reshape((1,))  # Poisson's ratio
        yield_stress = np.asarray(np.random.uniform(yield_stress_range[0], yield_stress_range[1]),
                                  dtype=DTYPE_NP).reshape((1,))  # Yield stress

        set_parameters(mpm_env, E, nu, yield_stress)

        forward_backward(mpm_env, init_state, trajectory.copy(), backward=True, render=False,
                         render_init_pcd=False, render_end_pcd=False, render_heightmap=False,
                         init_pcd_path=os.path.join(training_data_path, 'pcd_' + str(data_ind)+str(0) + '.ply'),
                         init_pcd_offset=env.pcd_offset,
                         init_mesh_path=env.mesh_file,
                         init_mesh_pos=env.initial_pos)

        print(f'===> CPU memory occupied after forward-backward: {process.memory_percent()} %')
        print(f'===> GPU memory after forward-backward: {get_gpu_memory()}')
        print('===> Gradients:')
        print(f"Gradient of E: {mpm_env.simulator.particle_param.grad[material_id].E}")
        print(f"Gradient of nu: {mpm_env.simulator.particle_param.grad[material_id].nu}")
        print(f"Gradient of rho: {mpm_env.simulator.particle_param.grad[material_id].rho}")
        print(f"Gradient of yield stress: {mpm_env.simulator.system_param.grad[None].yield_stress}")
        print(f"Gradient of manipulator friction: {mpm_env.simulator.system_param.grad[None].manipulator_friction}")
        print(f"Gradient of ground friction: {mpm_env.simulator.system_param.grad[None].ground_friction}")
        print(f"Gradient of theta_c: {mpm_env.simulator.system_param.grad[None].theta_c}")
        print(f"Gradient of theta_s: {mpm_env.simulator.system_param.grad[None].theta_s}")
        grad = np.array([mpm_env.simulator.particle_param.grad[material_id].E,
                         mpm_env.simulator.particle_param.grad[material_id].nu,
                         mpm_env.simulator.system_param.grad[None].yield_stress,
                         mpm_env.simulator.system_param.grad[None].manipulator_friction,
                         mpm_env.simulator.system_param.grad[None].ground_friction], dtype=DTYPE_NP)
        avg_grads += grad.copy()

    mpm_env.simulator.clear_ckpt()

avg_grads /= 45
print('===> Avg. Gradients:')
print(f"Avg. gradient of E: {avg_grads[0]}")
print(f"Avg. gradient of nu: {avg_grads[1]}")
print(f"Avg. gradient of yield stress: {avg_grads[2]}")
print(f"Avg. gradient of manipulator friction: {avg_grads[3]}")
print(f"Avg. gradient of ground friction: {avg_grads[4]}")
n = 0
while True:
    grad_file_name = os.path.join(gradient_file_path, f'grads-{str(n)}.npy')
    config_file_name = os.path.join(gradient_file_path, f'loss-config-{str(n)}.json')
    if not os.path.exists(grad_file_name):
        break
    n += 1

np.save(grad_file_name, avg_grads)
with open(os.path.join(gradient_file_path, f'loss-config-{str(n)}.json'), 'w') as f_ac:
    json.dump(loss_cfg, f_ac)
