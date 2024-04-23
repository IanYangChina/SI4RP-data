import numpy as np
import os
import taichi as ti
from time import time, sleep
import open3d as o3d
from vedo import Points, show, Mesh
from PIL import Image
import imageio
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

from doma.engine.utils.misc import get_gpu_memory
import psutil
import json
import argparse

DTYPE_NP = np.float32
DTYPE_TI = ti.f32
script_path = os.path.dirname(os.path.realpath(__file__))

from doma.envs.sys_id_env import make_env, set_parameters


def forward(mpm_env, init_state, trajectory, n_episode=-1, press_to_proceed=False, eval=False,
            render=False, save_img=False, save_heightmap=False, image_dir=None, save_loss=True, save_gif=False,
            render_init_pcd=False, render_end_pcd=False, render_heightmap=False,
            init_pcd_path=None, init_pcd_offset=None, init_mesh_path=None, init_mesh_pos=None):

    mpm_env.set_state(init_state['state'], grad_enabled=False)
    # print(mpm_env.agent.effectors[0].pos[mpm_env.simulator.cur_substep_local])
    if render_init_pcd:
        x_init, _ = mpm_env.render(mode='point_cloud')
        RGBA = np.zeros((len(x_init), 4))
        RGBA[:, 0] = x_init[:, 2] / x_init[:, 2].max() * 255
        RGBA[:, 1] = x_init[:, 2] / x_init[:, 2].max() * 255
        RGBA[:, -1] = 255
        pts_init = Points(x_init, r=12, c=RGBA)

        real_pcd_pts_init = Points(init_pcd_path, r=12, c='r')
        pcd_init = real_pcd_pts_init.points() + init_pcd_offset
        pcd_init[:, 1] += 0.1
        RGBA = np.zeros((len(pcd_init), 4))
        RGBA[:, -1] = 255
        RGBA[:, 0] = pcd_init[:, 2] / pcd_init[:, 2].max() * 255
        RGBA[:, 1] = pcd_init[:, 2] / pcd_init[:, 2].max() * 255
        real_pcd_pts_init = Points(pcd_init, r=12, c=RGBA)

        mesh_init = Mesh(init_mesh_path)
        coords_init = mesh_init.points()
        coords_init += init_mesh_pos
        coords_init[:, 1] += 0.2
        mesh_init.points(coords_init)

    if save_img or save_heightmap or save_loss or save_gif:
        img_dir = os.path.join(image_dir, str(n_episode))
        os.makedirs(img_dir, exist_ok=True)
        if not eval:
            frames_to_save = [
                # 0,
                # round(mpm_env.horizon/4),
                # round(mpm_env.horizon/2),
                # round(mpm_env.horizon*3/4),
                mpm_env.horizon - 1
            ]
        else:
            interval = mpm_env.horizon // 10
            frames_to_save = [0, interval, 2 * interval, 3 * interval, 4 * interval, 5 * interval,
                              6 * interval, 7 * interval, 8 * interval, 9 * interval, mpm_env.horizon - 1]
        if save_gif:
            frames_to_save = list(range(mpm_env.horizon))
        frames = []

    for i in range(mpm_env.horizon):
        action = trajectory[i]
        mpm_env.step(action)
        # print(mpm_env.agent.effectors[0].pos[mpm_env.simulator.cur_substep_local])
        if save_img or save_gif:
            if i in frames_to_save:
                img = mpm_env.render(mode='rgb_array')
                if eval or save_gif:
                    frames.append(img)
                else:
                    Image.fromarray(img).save(os.path.join(img_dir, f'img_{i}.png'))

        if render:
            mpm_env.render(mode='human')
        if press_to_proceed and i == 0:
            input('Press any key to proceed')

    loss_info = mpm_env.get_final_loss()
    for i, v in loss_info.items():
        if i != 'final_height_map':
            print(f'===> {i}: {v:.4f}')
        else:
            pass
    if save_loss:
        loss_info_to_save = loss_info.copy()
        loss_info_to_save['final_height_map'] = None
        with open(os.path.join(img_dir, 'loss_info.json'), 'w') as f:
            json.dump(loss_info_to_save, f, indent=4)

    if save_gif:
        with imageio.get_writer(os.path.join(img_dir, f'video.gif'), mode='I') as writer:
            for i in range(mpm_env.horizon):
                if i % 5 != 0:
                    continue
                writer.append_data(frames[i])

    if save_img and eval:
        fig, axes = plt.subplots(1, len(frames_to_save), figsize=(len(frames_to_save) * 2, 2))
        plt.subplots_adjust(wspace=0.01)
        for i in range(len(frames_to_save)):
            img = frames[i]
            axes[i].imshow(img)
            # axes[i].set_xlabel(f't{img_id}'.translate(SUB))
            axes[i].get_xaxis().set_visible(False)
            axes[i].get_yaxis().set_visible(False)
            axes[i].set_frame_on(False)
        # plt.tight_layout()
        plt.savefig(os.path.join(img_dir, f'img_combine.pdf'), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    if render_heightmap:
        cmap = 'YlOrBr'
        target_hm = mpm_env.loss.height_map_pcd_target.to_numpy()
        min_val, max_val = np.amin(target_hm), np.amax(target_hm)

        plt.imshow(loss_info['final_height_map'], cmap=cmap, vmin=min_val, vmax=max_val)
        plt.xticks([])
        plt.yticks([])

        if save_heightmap:
            plt.savefig(os.path.join(img_dir, 'end_heightmap.png'), bbox_inches='tight', dpi=300)
        else:
            plt.show()

    if render_end_pcd:
        y_offset = 0.0
        if render_init_pcd:
            y_offset = 0.3
        x, _ = mpm_env.render(mode='point_cloud')
        x[:, 1] += y_offset
        RGBA = np.zeros((len(x), 4))
        RGBA[:, -1] = 255
        RGBA[:, 0] = 250  # x[:, 0] / x[:, 0].max() * 255
        # RGBA[:, 0] = x[:, 1] / x[:, 1].max() * 255
        RGBA[:, 2] = x[:, 2] / x[:, 2].max() * 255
        pts = Points(x, r=12, c=RGBA)

        x_ = mpm_env.loss.target_pcd_points.to_numpy() / 1000
        x_[:, 1] += (y_offset + 0.1)
        RGBA = np.zeros((mpm_env.loss.n_target_pcd_points, 4))
        RGBA[:, -1] = 255
        RGBA[:, 0] = 250  # x_[:, 0] / x_[:, 0].max() * 255
        # RGBA[:, 0] = x_[:, 1] / x_[:, 1].max() * 255
        RGBA[:, 2] = x_[:, 2] / x_[:, 2].max() * 255
        real_pcd_pts = Points(x_, r=12, c=RGBA)

        x__ = mpm_env.loss.target_particles_from_mesh.to_numpy() / 1000
        x__[:, 1] += (y_offset + 0.2)
        RGBA = np.zeros((mpm_env.loss.n_target_particles_from_mesh, 4))
        RGBA[:, -1] = 255
        RGBA[:, 0] = 250  # x__[:, 0] / x__[:, 0].max() * 255
        # RGBA[:, 0] = x__[:, 1] / x__[:, 1].max() * 255
        RGBA[:, 2] = x__[:, 2] / x__[:, 2].max() * 255
        real_particle_pts = Points(x__, r=12, c=RGBA)

        mesh = Mesh(mpm_env.loss.target_mesh_path)
        coords = mesh.points()
        coords += mpm_env.loss.target_mesh_offset
        coords += init_mesh_pos
        coords[:, 1] += (y_offset + 0.3)
        mesh.points(coords)

        if render_init_pcd:
            show([pts_init, real_pcd_pts_init, mesh_init,
                  pts, real_pcd_pts, real_particle_pts, mesh], __doc__, axes=True).close()
            del pts_init, real_pcd_pts_init, mesh_init, x_init, pcd_init, coords_init
        else:
            show([pts, real_pcd_pts, real_particle_pts, mesh], __doc__, axes=True).close()
        del x, _, x_, x__, pts, real_pcd_pts, real_particle_pts, mesh, RGBA, coords


def main(args):
    process = psutil.Process(os.getpid())
    script_path = os.path.dirname(os.path.realpath(__file__))
    cam_cfg = {
        'pos': (0.4, 0.1, 0.1),
        'lookat': (0.25, 0.25, 0.03),
        'fov': 30,
        'lights': [{'pos': (0.5, 0.25, 0.2), 'color': (0.6, 0.6, 0.6)},
                   {'pos': (0.5, 0.5, 1.0), 'color': (0.6, 0.6, 0.6)},
                   {'pos': (0.5, 0.0, 1.0), 'color': (0.8, 0.8, 0.8)}],
        'particle_radius': 0.002,
        'res': (640, 640)
    }

    cam_cfg_cylinder = {
        'pos': (0.40, 0.07, 0.08),
        'lookat': (0.25, 0.25, 0.05),
        'fov': 30,
        'lights': [{'pos': (0.5, 0.25, 0.2), 'color': (0.6, 0.6, 0.6)},
                   {'pos': (0.5, 0.5, 1.0), 'color': (0.6, 0.6, 0.6)},
                   {'pos': (0.5, 0.0, 1.0), 'color': (0.8, 0.8, 0.8)}],
        'particle_radius': 0.002,
        'res': (640, 640)
    }
    cam_cfg_rectangle = {
        'pos': (0.46, 0.1, 0.067),
        'lookat': (0.25, 0.25, 0.06),
        'fov': 30,
        'lights': [{'pos': (0.5, 0.25, 0.2), 'color': (0.6, 0.6, 0.6)},
                   {'pos': (0.5, 0.5, 1.0), 'color': (0.6, 0.6, 0.6)},
                   {'pos': (0.5, 0.0, 1.0), 'color': (0.8, 0.8, 0.8)}],
        'particle_radius': 0.002,
        'res': (640, 640)
    }
    cam_cfg_round = {
        'pos': (0.45, 0.17, 0.07),
        'lookat': (0.25, 0.25, 0.055),
        'fov': 30,
        'lights': [{'pos': (0.5, 0.25, 0.2), 'color': (0.6, 0.6, 0.6)},
                   {'pos': (0.5, 0.5, 1.0), 'color': (0.6, 0.6, 0.6)},
                   {'pos': (0.5, 0.0, 1.0), 'color': (0.8, 0.8, 0.8)}],
        'particle_radius': 0.002,
        'res': (640, 640)
    }
    agent_colors = {
        'rectangle': (0.9, 0.1, 0.1, 1.0),
        'round': (0.8, 0.8, 0.8, 1.0),
        'cylinder': (0.2, 0.2, 0.2, 1.0)
    }

    p_density = args['ptcl_density']
    loss_cfg = {
        'exponential_distance': True,
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

    assert args['agent_ind'] in [0, 1, 2]
    agents = ['rectangle', 'round', 'cylinder']
    agent = agents[args['agent_ind']]
    if agent == 'rectangle':
        if args['eval']:
            cam_cfg = cam_cfg_rectangle
        agent_init_euler = (0, 0, 45)
    else:
        agent_init_euler = (0, 0, 0)

    if args['eval']:
        if agent == 'cylinder':
            cam_cfg = cam_cfg_cylinder
        elif agent == 'round':
            cam_cfg = cam_cfg_round

    motion_ind = str(args['motion_ind'])
    if not args['dt_avg']:
        dt_global = args['dt']
        trajectory = np.load(
            os.path.join(script_path, '..', 'trajectories', f'tr_{motion_ind}_v_dt_{dt_global:0.2f}.npy'))
    else:
        dt_global = np.load(os.path.join(script_path, '..', 'trajectories', f'tr_{motion_ind}_dt_avg.npy'))
        trajectory = np.load(os.path.join(script_path, '..', 'trajectories', f'tr_{motion_ind}_v_dt_avg.npy'))
    n_substeps = 50

    if args['demo']:
        trajectory = np.load(os.path.join(script_path, '..', 'demo_files', 'eef_v_trajectory_test.npy'))
        dt_global = 0.003
        n_substeps = 10

    if args['eval']:
        if not args['dt_avg']:
            dt_global = args['dt']
            trajectory = np.load(
                os.path.join(script_path, '..', 'trajectories', f'tr_valid_{agent}_v_dt_{dt_global:0.2f}.npy'))
        else:
            dt_global = np.load(os.path.join(script_path, '..', 'trajectories', f'tr_valid_{agent}_dt_avg.npy'))
            trajectory = np.load(os.path.join(script_path, '..', 'trajectories', f'tr_valid_{agent}_v_dt_avg.npy'))

    horizon = trajectory.shape[0]
    training_data_path = os.path.join(script_path, '..', f'data-motion-{motion_ind}', f'eef-{agent}')
    if args['eval']:
        training_data_path = os.path.join(script_path, '..', 'data-motion-validation', f'eef-{agent}')

    E = 1e4  # [1e4, 3e5]
    nu = 0.3  # [0.01, 0.48]
    yield_stress = 6e3  # [1e2, 2e4]
    rho = 1300  # [1000, 2000]
    gf = 2.0  # [0.01, 2.0]
    mf = 0.3  # [0.01, 2.0]
    image_dir = None
    if args['load_params']:
        run_id = args['load_params_run']
        seed_id = args['load_params_seed']
        p_set = args['param_set']
        if args['oneshot']:
            assert not args['fewshot'], 'Cannot load oneshot and fewshot parameters at the same time'
            print(f'Loading optimised parameters from oneshot result, param set {p_set}, run {run_id} seed {seed_id}...')
            data_dir = os.path.join(script_path, '..',
                                    f'optimisation-oneshot-param{p_set}-run{run_id}-logs',
                                    f'seed-{seed_id}')
            save_dir = os.path.join(script_path, '..',
                                    f'optimisation-oneshot-param{p_set}-result-figs',
                                    f'run{run_id}', f'seed{seed_id}')
        elif args['fewshot']:
            assert not args['oneshot'], 'Cannot load oneshot and fewshot parameters at the same time'
            print(f'Loading optimised parameters from fewshot result, param set {p_set}, run {run_id} seed {seed_id}...')
            data_dir = os.path.join(script_path, '..',
                                    f'optimisation-fewshot-param{p_set}-run{run_id}-logs',
                                    f'seed-{seed_id}')
            save_dir = os.path.join(script_path, '..',
                                    f'optimisation-fewshot-param{p_set}-result-figs',
                                    f'run{run_id}', f'seed{seed_id}')
        elif args['realoneshot']:
            assert not args['oneshot'] and not args['fewshot'], \
                'Cannot load oneshot and fewshot parameters at the same time'
            realoneshot_agent = agents[args['realoneshot_agent_ind']]
            print(f'Loading optimised parameters from realoneshot result, '
                  f'param set {p_set}, agent {realoneshot_agent}, run {run_id} seed {seed_id}...')
            data_dir = os.path.join(script_path, '..',
                                    f'optimisation-realoneshot-{realoneshot_agent}-param{p_set}-run{run_id}-logs',
                                    f'seed-{seed_id}')
            save_dir = os.path.join(script_path, '..',
                                    f'optimisation-realoneshot-{realoneshot_agent}-param{p_set}-result-figs',
                                    f'run{run_id}', f'seed{seed_id}')
        else:
            raise ValueError('Please specify either oneshot or fewshot')

        params = np.load(os.path.join(data_dir, 'final_params.npy')).flatten()
        E = params[0]
        nu = params[1]
        yield_stress = params[2]
        rho = params[3]
        if p_set == 1:
            mf = params[4]
            gf = params[5]
        if args['img_dir'] is None:
            image_dir = os.path.join(save_dir, f'validation_tr_imgs-motion{motion_ind}-{agent}')
            if args['eval']:
                image_dir = os.path.join(save_dir, f'validation_tr_imgs-long_motion-{agent}')
        else:
            image_dir = os.path.join(script_path, '..', args['img_dir'])

    validation_dataind_dict = {
        '2': {
            'rectangle': [4, 8],
            'round': [4, 7],
            'cylinder': [4, 7]
        },
        '4': {
            'rectangle': [2, 4],
            'round': [2, 3],
            'cylinder': [1, 3]
        }
    }

    if args['eval']:
        data_ids = [0, 1]
    else:
        data_ids = validation_dataind_dict[motion_ind][agent]
    n_episode = 0
    for data_ind in data_ids:
        ti.reset()
        ti.init(arch=ti.cuda, default_fp=DTYPE_TI, default_ip=ti.i32, debug=False, device_memory_GB=3,
                fast_math=True, advanced_optimization=True, random_seed=1)
        data_cfg = {
            'data_path': training_data_path,
            'data_ind': str(data_ind),
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
        mpm_env.agent.effectors[0].mesh.update_color(agent_colors[agent])
        print(f'===> Num. simulation particles: {mpm_env.loss.n_particles_matching_mat}')
        print(f'===> Num. target pcd points: {mpm_env.loss.n_target_pcd_points}')
        print(f'===> Num. target particles: {mpm_env.loss.n_target_particles_from_mesh}')
        print(f'===> CPU memory occupied after create env: {process.memory_percent()} %')
        print(f'===> GPU memory after create env: {get_gpu_memory()}')

        print(f'===> Parameters: E = {E}, nu = {nu}, yield_stress = {yield_stress}, rho = {rho}, gf = {gf}, mf = {mf}')
        set_parameters(mpm_env, env_cfg['material_id'], E, nu, yield_stress,
                       rho=rho, ground_friction=gf, manipulator_friction=mf)
        forward(mpm_env, init_state, trajectory.copy(), n_episode=n_episode,
                press_to_proceed=args['press_to_proceed'], eval=args['eval'], save_loss=args['save_loss'], save_gif=args['save_gif'],
                render=args['render_human'], save_img=args['save_img'], save_heightmap=args['save_heightmap'], image_dir=image_dir,
                render_init_pcd=args['render_init_pcd'],
                render_end_pcd=args['render_end_pcd'], render_heightmap=args['render_heightmap'],
                init_pcd_path=os.path.join(training_data_path, 'pcd_' + str(data_ind) + str(0) + '.ply'),
                init_pcd_offset=env.target_pcd_offset,
                init_mesh_path=env.mesh_file,
                init_mesh_pos=env.initial_pos)

        print(f'===> CPU memory occupied after forward: {process.memory_percent()} %')
        print(f'===> GPU memory after forward: {get_gpu_memory()}')
        n_episode += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=float, default=4e7)
    parser.add_argument('--dsvs', dest='down_sample_voxel_size', type=float, default=0.006)
    parser.add_argument('--ptp', dest='press_to_proceed', default=False, action='store_true')
    parser.add_argument('--demo', dest='demo', default=False, action='store_true')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true')
    parser.add_argument('--load_params', dest='load_params', default=False, action='store_true')
    parser.add_argument('--load_params_run', dest='load_params_run', type=int, default=0)
    parser.add_argument('--load_params_seed', dest='load_params_seed', type=int, default=0)
    parser.add_argument('--fewshot', dest='fewshot', default=False, action='store_true')
    parser.add_argument('--oneshot', dest='oneshot', default=False, action='store_true')
    parser.add_argument('--realoneshot', dest='realoneshot', default=False, action='store_true')
    parser.add_argument('--realoneshot_agent_ind', dest='realoneshot_agent_ind', type=int, default=0)
    parser.add_argument('--param_set', dest='param_set', type=int, default=0)
    parser.add_argument('--dt', dest='dt', type=float, default=0.01)
    parser.add_argument('--dt_avg', dest='dt_avg', default=False, action='store_true')
    parser.add_argument('--m_id', dest='motion_ind', type=int, default=1)
    parser.add_argument('--agent_ind', dest='agent_ind', type=int, default=0)
    parser.add_argument('--r_human', dest='render_human', default=False, action='store_true')
    parser.add_argument('--save_loss', dest='save_loss', default=False, action='store_true')
    parser.add_argument('--save_img', dest='save_img', default=False, action='store_true')
    parser.add_argument('--save_heightmap', dest='save_heightmap', default=False, action='store_true')
    parser.add_argument('--r_init_pcd', dest='render_init_pcd', default=False, action='store_true')
    parser.add_argument('--r_end_pcd', dest='render_end_pcd', default=False, action='store_true')
    parser.add_argument('--r_hm', dest='render_heightmap', default=False, action='store_true')
    parser.add_argument('--img_dir', dest='img_dir', type=str, default=None)
    parser.add_argument('--save_gif', dest='save_gif', default=False, action='store_true')
    args = vars(parser.parse_args())
    main(args)
