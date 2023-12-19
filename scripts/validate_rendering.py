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
script_path = os.path.dirname(os.path.realpath(__file__))

from doma.envs.sys_id_env import make_env, set_parameters

def forward_backward(mpm_env, init_state, trajectory,
                     render=False, save_img=False, render_init_pcd=False, render_end_pcd=False, render_heightmap=False,
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

    if save_img:
        k = 0
        while True:
            img_dir = os.path.join(script_path, '..', 'demo_files', f'imgs-{k}')
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
                break
            k += 1
    t1 = time()
    mpm_env.set_state(init_state['state'], grad_enabled=True)
    # print(mpm_env.agent.effectors[0].pos[mpm_env.simulator.cur_substep_local])
    for i in range(mpm_env.horizon):
        action = trajectory[i]
        mpm_env.step(action)
        # print(mpm_env.agent.effectors[0].pos[mpm_env.simulator.cur_substep_local])
        if save_img:
            frame_skip = 4
            if i % frame_skip == 0:
                img = mpm_env.render(mode='rgb_array')
                np.save(os.path.join(img_dir, f'img_{i//frame_skip}.npy'), img)
                # plt.imshow(img)
                # plt.show()
                # sleep(0.1)
        if render:
            mpm_env.render(mode='human')
        if i == 0:
            input('Press any key to proceed')

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


def main(args):
    process = psutil.Process(os.getpid())
    script_path = os.path.dirname(os.path.realpath(__file__))
    gradient_file_path = os.path.join(script_path, '..', 'gradients')
    os.makedirs(gradient_file_path, exist_ok=True)

    cam_cfg = {
        'pos': (0.2, 0.02, 0.07),
        'lookat': (0.24, 0.23, 0.07),
        'fov': 30,
        'lights': [{'pos': (0.5, -1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
                   {'pos': (0.5, -1.5, 1.5), 'color': (0.5, 0.5, 0.5)}]
    }

    p_density = args['ptcl_density']
    loss_cfg = {
        'exponential_distance': False,
        'averaging_loss': False,
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
        'emd_point_distance_loss': False,
        'emd_particle_distance_loss': False
    }

    motion_ind = str(args['motion_ind'])
    trajectory = np.load(os.path.join(script_path, '..', f'data-motion-{motion_ind}', 'tr_eef_v.npy'))
    dt_global = np.load(os.path.join(script_path, '..', f'data-motion-{motion_ind}', 'tr_dt.npy'))
    horizon = trajectory.shape[0]
    n_substeps = 50

    if args['demo']:
        trajectory = np.load(os.path.join(script_path, '..', 'demo_files', 'eef_v_trajectory_test.npy'))
        horizon = 800
        dt_global = 0.003
        n_substeps = 10

    assert args['agent_ind'] in [0, 1, 2]
    agents = ['rectangle', 'round', 'cylinder']
    agent = agents[args['agent_ind']]
    if agent == 'rectangle':
        agent_init_euler = (0, 0, 45)
    else:
        agent_init_euler = (0, 0, 0)
    training_data_path = os.path.join(script_path, '..', f'data-motion-{motion_ind}', f'eef-{agent}')
    data_ids = ['2', '3', '4']

    E = 38400
    nu = 0.40
    yield_stress = 500

    for data_ind in data_ids:
        ti.reset()
        ti.init(arch=ti.opengl, default_fp=DTYPE_TI, default_ip=ti.i32,
                fast_math=False, random_seed=1)
        data_cfg = {
            'data_path': training_data_path,
            'data_ind': data_ind,
        }
        env_cfg = {
            'p_density': p_density,
            'horizon': horizon,
            'dt_global': dt_global,
            'n_substeps': n_substeps,
            'material_id': 2,
            'agent_name': agent,
            'agent_init_euler': agent_init_euler,
        }
        print(f'===> CPU memory occupied before create env: {process.memory_percent()} %')
        print(f'===> GPU memory before create env: {get_gpu_memory()}')
        env, mpm_env, init_state = make_env(data_cfg, env_cfg, loss_cfg, cam_cfg)
        print(f'===> Num. simulation particles: {mpm_env.loss.n_particles_matching_mat}')
        print(f'===> Num. target pcd points: {mpm_env.loss.n_target_pcd_points}')
        print(f'===> Num. target particles: {mpm_env.loss.n_target_particles_from_mesh}')
        print(f'===> CPU memory occupied after create env: {process.memory_percent()} %')
        print(f'===> GPU memory after create env: {get_gpu_memory()}')

        set_parameters(mpm_env, env_cfg['material_id'], E, nu, yield_stress,
                       rho=1000, ground_friction=0.5, manipulator_friction=0.5)
        forward_backward(mpm_env, init_state, trajectory.copy(),
                         render=args['render_human'], save_img=args['save_img'],
                         render_init_pcd=args['render_init_pcd'],
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
    parser.add_argument('--demo', dest='demo', default=False, action='store_true')
    parser.add_argument('--m_id', dest='motion_ind', type=int, default=1)
    parser.add_argument('--agent_ind', dest='agent_ind', type=int, default=0)
    parser.add_argument('--r_human', dest='render_human', default=False, action='store_true')
    parser.add_argument('--save_img', dest='save_img', default=False, action='store_true')
    parser.add_argument('--r_init_pcd', dest='render_init_pcd', default=False, action='store_true')
    parser.add_argument('--r_end_pcd', dest='render_end_pcd', default=False, action='store_true')
    parser.add_argument('--r_hm', dest='render_heightmap', default=False, action='store_true')
    args = vars(parser.parse_args())
    main(args)
