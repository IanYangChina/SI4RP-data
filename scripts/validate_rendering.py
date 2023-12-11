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
import argparse

DTYPE_NP = np.float32
DTYPE_TI = ti.f32

from doma.envs import SysIDEnv

def forward_backward(mpm_env, init_state, trajectory,
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
        del x, pts, real_pcd_pts, mesh, RGBA, x_, coords

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
        del fig, ax1, ax2, im1, im2, cax, divider

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
        del x, _, x_, x__, pts, real_pcd_pts, real_particle_pts, mesh, RGBA, coords


def make_env(data_path, data_ind, horizon, dt_global, agent_name, material_id, cam_cfg, loss_config):
    obj_start_mesh_file_path = os.path.join(data_path, 'mesh_' + data_ind + str(0) + '_repaired_normalised.obj')
    if not os.path.exists(obj_start_mesh_file_path):
        return None, None
    obj_start_centre_real = np.load(
        os.path.join(data_path, 'mesh_' + data_ind + str(0) + '_repaired_centre.npy')).astype(DTYPE_NP)
    obj_start_centre_top_normalised = np.load(
        os.path.join(data_path, 'mesh_' + data_ind + str(0) + '_repaired_normalised_centre_top.npy')).astype(DTYPE_NP)

    obj_end_pcd_file_path = os.path.join(data_path, 'pcd_' + data_ind + str(1) + '.ply')
    obj_end_mesh_file_path = os.path.join(data_path, 'mesh_' + data_ind + str(1) + '_repaired_normalised.obj')
    obj_end_centre_top_normalised = np.load(
        os.path.join(data_path, 'mesh_' + data_ind + str(1) + '_repaired_normalised_centre_top.npy')).astype(DTYPE_NP)

    # Building environment
    obj_start_initial_pos = np.array([0.25, 0.25, obj_start_centre_top_normalised[-1] + 0.01], dtype=DTYPE_NP)
    agent_init_pos = (0.25, 0.25, 2 * obj_start_centre_top_normalised[-1] + 0.01)
    height_map_res = loss_config['height_map_res']
    loss_config.update({
        'target_pcd_path': obj_end_pcd_file_path,
        'pcd_offset': (-obj_start_centre_real + obj_start_initial_pos),
        'target_mesh_file': obj_end_mesh_file_path,
        'mesh_offset': (0.25, 0.25, obj_end_centre_top_normalised[-1] + 0.01),
        'target_pcd_height_map_path': os.path.join(data_path,
                                                   f'target_pcd_height_map-{data_ind}-res{str(height_map_res)}-vdsize{str(0.001)}.npy'),
    })

    env = SysIDEnv(ptcl_density=loss_config['ptcl_density'], horizon=horizon, dt_global=dt_global,
                   material_id=material_id, voxelise_res=1080,
                   mesh_file=obj_start_mesh_file_path, initial_pos=obj_start_initial_pos,
                   loss_cfg=loss_config,
                   agent_cfg_file=agent_name + '_eef.yaml', agent_init_pos=agent_init_pos, agent_init_euler=(0, 0, 0),
                   render_agent=True, camera_cfg=cam_cfg)
    env.reset()
    mpm_env = env.mpm_env
    init_state = mpm_env.get_state()

    return env, mpm_env, init_state


def set_parameters(mpm_env, E, nu, yield_stress):
    mpm_env.simulator.system_param[None].yield_stress = yield_stress
    mpm_env.simulator.particle_param[2].E = E
    mpm_env.simulator.particle_param[2].nu = nu
    mpm_env.simulator.particle_param[2].rho = 1300


def main(args):
    process = psutil.Process(os.getpid())
    script_path = os.path.dirname(os.path.realpath(__file__))
    gradient_file_path = os.path.join(script_path, '..', 'gradients')
    os.makedirs(gradient_file_path, exist_ok=True)

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

    p_density = args['ptcl_density']
    loss_cfg = {
        'exponential_distance': False,
        'averaging_loss': True,
        'point_distance_rs_loss': False,
        'point_distance_sr_loss': False,
        'down_sample_voxel_size': args['down_sample_voxel_size'],
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

    moition_ind = str(args['motion_ind'])
    trajectory = np.load(os.path.join(script_path, '..', f'data-motion-{moition_ind}', 'eef_v_trajectory_.npy'))
    if moition_ind == '1':
        horizon = 150
        dt_global = 1.03 / trajectory.shape[0]
    else:
        horizon = 200
        dt_global = 1.04 / trajectory.shape[0]

    assert args['agent_ind'] in [0, 1, 2]
    agents = ['rectangle','round', 'cylinder']
    agent = agents[args['agent_ind']]
    training_data_path = os.path.join(script_path, '..', f'data-motion-{moition_ind}', f'eef-{agent}')
    data_ids = ['2', '4', '8']

    E = 68400
    nu = 0.49
    yield_stress = 702

    for data_ind in data_ids:
        ti.reset()
        ti.init(arch=ti.opengl, default_fp=DTYPE_TI, default_ip=ti.i32,
                fast_math=False, random_seed=1)
        print(f'===> CPU memory occupied before create env: {process.memory_percent()} %')
        print(f'===> GPU memory before create env: {get_gpu_memory()}')
        env, mpm_env, init_state = make_env(training_data_path, str(data_ind), horizon, dt_global, agent,
                                            material_id, cam_cfg, loss_cfg.copy())
        print(f'===> Num. simulation particles: {mpm_env.loss.n_particles_matching_mat}')
        print(f'===> Num. target pcd points: {mpm_env.loss.n_target_pcd_points}')
        print(f'===> Num. target particles: {mpm_env.loss.n_target_particles_from_mesh}')
        print(f'===> CPU memory occupied after create env: {process.memory_percent()} %')
        print(f'===> GPU memory after create env: {get_gpu_memory()}')

        set_parameters(mpm_env, E, nu, yield_stress)
        forward_backward(mpm_env, init_state, trajectory.copy(),
                         render=args['render_human'], render_init_pcd=args['render_init_pcd'],
                         render_end_pcd=args['render_end_pcd'], render_heightmap=args['render_heightmap'],
                         init_pcd_path=os.path.join(training_data_path, 'pcd_' + str(data_ind) + str(0) + '.ply'),
                         init_pcd_offset=env.pcd_offset,
                         init_mesh_path=env.mesh_file,
                         init_mesh_pos=env.initial_pos)

        print(f'===> CPU memory occupied after forward: {process.memory_percent()} %')
        print(f'===> GPU memory after forward: {get_gpu_memory()}')

        mpm_env.simulator.clear_ckpt()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=float, default=3e7)
    parser.add_argument('--dsvs', dest='down_sample_voxel_size', type=float, default=0.004)
    parser.add_argument('--m_id', dest='motion_ind', type=int, default=1)
    parser.add_argument('--agent_ind', dest='agent_ind', type=int, default=0)
    parser.add_argument('--r_human', dest='render_human', default=False, action='store_true')
    parser.add_argument('--r_init_pcd', dest='render_init_pcd', default=False, action='store_true')
    parser.add_argument('--r_end_pcd', dest='render_end_pcd', default=False, action='store_true')
    parser.add_argument('--r_hm', dest='render_heightmap', default=False, action='store_true')
    args = vars(parser.parse_args())
    main(args)
