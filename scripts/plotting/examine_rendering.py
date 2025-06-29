import numpy as np
import os
import taichi as ti
from vedo import Points, show, Mesh
from PIL import Image
import imageio
import matplotlib as mpl
import matplotlib.pylab as plt
import time

import psutil
import json
import argparse

DTYPE_NP = np.float32
DTYPE_TI = ti.f32
script_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_path, '..')

from doma.envs.sys_id_env import make_env
from doma.engine.utils.misc import get_gpu_memory, set_parameters
from doma.engine.renderer.plb_renderer import Renderer, RENDERER
RENDERER.max_ray_depth = 2
RENDERER.image_res = (512, 512)
RENDERER.ground_color = [0.9, 0.9, 0.9]
RENDERER.camera_pos = (0.25, 0.35, 0.6)
RENDERER.camera_rot = (0.7, 0)
RENDERER.use_directional_light = True
RENDERER.light_direction = (1., 1., 0.7)
# Red = 76, Green = 0, Blue = 153 → hex
particle_color = (76 << 16) + (0 << 8) + 153


def forward(mpm_env, init_state, trajectory, render=False,
            save_img=False, save_heightmap=False,  save_gif=False, save_tr_combined_img=False,
            img_dir=None, save_loss=True, use_ray_tracing_renderer=False,
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
        # img_dir = os.path.join(img_dir, str(n_episode))
        os.makedirs(img_dir, exist_ok=True)
        interval = mpm_env.horizon // 10
        frames_to_save = [0, interval, 2 * interval, 3 * interval, 4 * interval, 5 * interval,
                          6 * interval, 7 * interval, 8 * interval, 9 * interval, mpm_env.horizon - 1]
        if save_gif:
            frames_to_save_for_gifs = list(range(mpm_env.horizon))

    if use_ray_tracing_renderer:
        plb_renderer = Renderer(cfg=RENDERER, primitives=[mpm_env.agent.effectors[0].mesh])
        if render:
            plt.ion()
            figure, ax = plt.subplots(figsize=(6, 6))
            plt.axis('off')

    frames = []
    frames_for_gifs = []
    for i in range(mpm_env.horizon):
        action = trajectory[i]
        mpm_env.step(action)
        if save_img or save_gif:
            cur_time = time.time()
            if use_ray_tracing_renderer:
                particle_np = mpm_env.simulator.get_x()
                particle_np_swap = np.zeros_like(particle_np)
                particle_np_swap[:, 0] = particle_np[:, 0]
                particle_np_swap[:, 1] = particle_np[:, 2]
                particle_np_swap[:, 2] = particle_np[:, 1]
                plb_renderer.set_particles(particle_np_swap, np.zeros(particle_np.shape[0], ) + particle_color)
                img = plb_renderer.render_frame(20, shape=1, primitive=1, target=0)
                img = (img * 255).astype(np.uint8)
            else:
                img = mpm_env.render(mode='rgb_array')
            print("Step %i of %i, Rendering FPS: %0.2f" % (i, mpm_env.horizon, 1 / (time.time() - cur_time)), end='\r')

            # Image.fromarray(img).save(os.path.join(img_dir, f'img_{i}.png'))
            if save_img and i in frames_to_save:
                frames.append(img)
            if save_gif and i in frames_to_save_for_gifs:
                frames_for_gifs.append(img)

        if render:
            if use_ray_tracing_renderer:
                particle_np = mpm_env.simulator.get_x()
                particle_np_swap = np.zeros_like(particle_np)
                particle_np_swap[:, 0] = particle_np[:, 0]
                particle_np_swap[:, 1] = particle_np[:, 2]
                particle_np_swap[:, 2] = particle_np[:, 1]
                plb_renderer.set_particles(particle_np_swap, np.zeros(particle_np.shape[0], ) + particle_color)
                img = plb_renderer.render_frame(20, shape=1, primitive=1, target=0)
                img = (img * 255).astype(np.uint8)
                plt.imshow(img)
                figure.canvas.draw()
                figure.canvas.flush_events()
            else:
                mpm_env.render(mode='human')

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
        n = 0
        done = False
        while not done:
            if os.path.exists(os.path.join(img_dir, f'video-{n}.gif')):
                n += 1
            else:
                done = True
        with imageio.get_writer(os.path.join(img_dir, f'video-{n}.gif'), mode='I') as writer:
            for i in range(mpm_env.horizon):
                if i % 5 != 0:
                    continue
                writer.append_data(frames_for_gifs[i])

    if save_img:
        mpl.use('Agg')
        fig, axes = plt.subplots(1, len(frames_to_save), figsize=(len(frames_to_save) * 2, 2))
        plt.subplots_adjust(wspace=0.01)
        for i in range(len(frames_to_save)):
            img = frames[i]
            axes[i].imshow(img)
            axes[i].get_xaxis().set_visible(False)
            axes[i].get_yaxis().set_visible(False)
            axes[i].set_frame_on(False)
        plt.savefig(os.path.join(img_dir, f'img_combine.pdf'), bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)

    if save_heightmap or render_heightmap:
        if save_heightmap:
            mpl.use('Agg')
        cmap = 'YlOrBr'
        target_hm = mpm_env.loss.height_map_pcd_target.to_numpy()
        min_val, max_val = np.amin(target_hm), np.amax(target_hm)

        plt.imshow(loss_info['final_height_map'], cmap=cmap, vmin=min_val, vmax=max_val)
        plt.xticks([])
        plt.yticks([])

        if render_heightmap:
            plt.show()
        if save_heightmap:
            plt.savefig(os.path.join(img_dir, 'end_heightmap.png'), bbox_inches='tight', pad_inches=0, dpi=300)

    if render_end_pcd:
        y_offset = 0.0
        if render_init_pcd:
            y_offset = 0.3
        x, _ = mpm_env.render(mode='point_cloud')
        x[:, 1] += y_offset
        RGBA = np.zeros((len(x), 4))
        RGBA[:, -1] = 255
        RGBA[:, 0] = 250
        RGBA[:, 2] = x[:, 2] / x[:, 2].max() * 255
        pts = Points(x, r=12, c=RGBA)

        x_ = mpm_env.loss.target_pcd_points.to_numpy() / 1000
        x_[:, 1] += (y_offset + 0.1)
        RGBA = np.zeros((mpm_env.loss.n_target_pcd_points, 4))
        RGBA[:, -1] = 255
        RGBA[:, 0] = 250
        RGBA[:, 2] = x_[:, 2] / x_[:, 2].max() * 255
        real_pcd_pts = Points(x_, r=12, c=RGBA)

        x__ = mpm_env.loss.target_particles_from_mesh.to_numpy() / 1000
        x__[:, 1] += (y_offset + 0.2)
        RGBA = np.zeros((mpm_env.loss.n_target_particles_from_mesh, 4))
        RGBA[:, -1] = 255
        RGBA[:, 0] = 250
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

    cam_cfg = {
        'pos': (0.4, 0.1, 0.1),
        'lookat': (0.25, 0.25, 0.03),
        'fov': 30,
        'lights': [{'pos': (0.5, 0.25, 0.2), 'color': (0.6, 0.6, 0.6)},
                   {'pos': (0.5, 0.5, 1.0), 'color': (0.6, 0.6, 0.6)},
                   {'pos': (0.5, 0.0, 1.0), 'color': (0.8, 0.8, 0.8)}],
        'particle_radius': 0.002,
        'res': (640, 640),
        'euler': (135, 0, 180),
        'focal_length': 0.01
    }

    agent_colors = {
        'rectangle': (0.9, 0.1, 0.1, 1.0),
        'round': (0.8, 0.8, 0.8, 1.0),
        'cylinder': (0.2, 0.2, 0.2, 1.0)
    }

    p_density = args['ptcl_density']
    loss_cfg = {
        'exponential_distance': False,
        'averaging_loss': False,
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
        'emd_point_distance_loss': False,
        'emd_particle_distance_loss': False
    }
    if args['load_params_dataset'] == 'soil':
        loss_cfg['height_map_size'] = 0.17
    if args['load_params_dataset'] == 'slime':
        loss_cfg['height_map_size'] = 0.15

    assert args['agent_ind'] in [0, 1, 2]
    agents = ['rectangle', 'round', 'cylinder']
    agent = agents[args['agent_ind']]
    if agent == 'rectangle':
        cam_cfg['pos'] = (0.46, 0.1, 0.067)
        cam_cfg['lookat'] = (0.25, 0.25, 0.06)
        agent_init_euler = (0, 0, 45)
    elif agent == 'cylinder':
        cam_cfg['pos'] = (0.40, 0.07, 0.08)
        cam_cfg['lookat'] = (0.25, 0.25, 0.05)
        agent_init_euler = (0, 0, 0)
        if args['load_params_dataset'] == 'soil':
            cam_cfg['pos'] = (0.365, 0.07, 0.06)
            if args['long_motion']:
                cam_cfg['pos'] = (0.355, 0.07, 0.05)
            cam_cfg['lookat'] = (0.25, 0.25, 0.03)
        else:
            # args['load_params_dataset'] == 'slime'
            cam_cfg['pos'] = (0.37, 0.04, 0.05)
            cam_cfg['lookat'] = (0.27, 0.25, 0.03)
            if args['long_motion']:
                cam_cfg['pos'] = (0.37, 0.06, 0.07)
                cam_cfg['lookat'] = (0.25, 0.25, 0.03)
    else:
        # agent == 'round':
        cam_cfg['pos'] = (0.45, 0.17, 0.07)
        cam_cfg['lookat'] = (0.25, 0.25, 0.055)
        agent_init_euler = (0, 0, 0)

    contact_level = args['contact_level']
    motion_name = 'poking' if contact_level == 1 else 'poking-shifting'
    motion_ind = str(2)
    if args['long_motion']:
        data_path = os.path.join(script_path, '..', 'data', 'data-motion-long-horizon', f'eef-{agent}')
        if args['load_params_dataset'] in ['slime', 'soil']:
            data_path = os.path.join(script_path, '..', 'data', 'other_mats', args['load_params_dataset'], 'long-motion-validation')
        if not args['dt_avg']:
            dt_global = args['dt']
            trajectory = np.load(os.path.join(script_path, '..', 'data', 'trajectories', f'tr_long-horizon-{agent}_v_dt_{dt_global:0.2f}.npy'))
        else:
            dt_global = np.load(os.path.join(script_path, '..', 'data', 'trajectories', f'tr_long-horizon-{agent}_dt_avg.npy'))
            trajectory = np.load(os.path.join(script_path, '..', 'data', 'trajectories', f'tr_long-horizon-{agent}_v_dt_avg.npy'))
    else:
        data_path = os.path.join(script_path, '..', 'data', f'data-motion-{motion_name}-{motion_ind}', f'eef-{agent}', 'validation_data')
        if args['load_params_dataset'] in ['slime', 'soil']:
            data_path = os.path.join(script_path, '..', 'data', 'other_mats', args['load_params_dataset'])
        if not args['dt_avg']:
            dt_global = args['dt']
            trajectory = np.load(os.path.join(script_path, '..', 'data', 'trajectories', f'tr_{motion_name}_{motion_ind}_v_dt_{dt_global:0.2f}.npy'))
        else:
            dt_global = np.load(os.path.join(script_path, '..', 'data', 'trajectories', f'tr_{motion_name}_{motion_ind}_dt_avg.npy'))
            trajectory = np.load(os.path.join(script_path, '..', 'data', 'trajectories', f'tr_{motion_name}_{motion_ind}_v_dt_avg.npy'))

    horizon = trajectory.shape[0]

    E = np.asarray([30000], dtype=DTYPE_NP).reshape((1,))
    nu = np.asarray([0.4], dtype=DTYPE_NP).reshape((1,))
    yield_stress = np.asarray([1000], dtype=DTYPE_NP).reshape((1,))
    rho = np.asarray([1330], dtype=DTYPE_NP).reshape((1,))
    mf = np.asarray([0.7], dtype=DTYPE_NP).reshape((1,))
    gf = np.asarray([0.7], dtype=DTYPE_NP).reshape((1,))
    save_dir = os.path.join(script_path, '..', 'optimisation-results', 'figures')
    if args['load_params']:
        run_id = args['load_params_run']
        seed_id = args['load_params_seed']
        dataset = args['load_params_dataset']
        assert dataset in ['12mix', '6mix', '1cyl', '1rec', '1round', 'slime', 'soil'], f"Dataset {dataset} not found."
        print(f'Loading parameters from optimisation result at contact level {contact_level} with dataset {dataset}, '
              f'run {run_id} seed {seed_id}...')
        data_dir = os.path.join(script_path, '..', 'optimisation-results',
                                f'level{contact_level}-{dataset}-run{run_id}-logs',
                                f'seed-{seed_id}')
        save_dir = os.path.join(script_path, '..', 'optimisation-results', 'figures',
                                f'level{contact_level}-{dataset}', f'run{run_id}', f'seed{seed_id}')

        params = np.load(os.path.join(data_dir, 'final_params.npy')).flatten()
        E = params[0]
        nu = params[1]
        yield_stress = params[2]
        rho = params[3]
        mf = np.asarray([0.3], dtype=DTYPE_NP).reshape((1,))
        gf = np.asarray([2.0], dtype=DTYPE_NP).reshape((1,))
        if contact_level == 2:
            mf = params[4]
            gf = params[5]
    print(f'===> Parameters: E = {E}, nu = {nu}, yield_stress = {yield_stress}, rho = {rho}, gf = {gf}, mf = {mf}')

    if args['img_dir'] is None:
        if args['long_motion']:
            image_dir = os.path.join(save_dir, f'long_motion-{agent}')
        else:
            image_dir = os.path.join(save_dir, f'motion{motion_ind}-{agent}')
    else:
        image_dir = os.path.join(save_dir, 'rendering-results', args['img_dir'])
    print(f"Images/videos will be saved in {image_dir}, if any.")

    data_ids = [0, 1]
    agent_init_pos_xy_offset = (0.0, 0.0)
    if args['load_params_dataset'] in ['slime', 'soil']:
        data_ids = [0]
        agent_init_pos_xy_offset = (0.01, 0.0)
    n_episode = 0
    for data_ind in data_ids:
        image_dir_ = os.path.join(image_dir, f'data-{data_ind}')
        ti.reset()
        ti.init(arch=ti.cuda, default_fp=DTYPE_TI, default_ip=ti.i32, debug=False, device_memory_GB=3,
                fast_math=True, advanced_optimization=True, random_seed=1)
        data_cfg = {
            'data_path': data_path,
            'data_ind': str(data_ind),
        }
        env_cfg = {
            'p_density': p_density,
            'horizon': horizon,
            'dt_global': dt_global,
            'n_substeps': 50,
            'material_id': 2,
            'agent_name': agent,
            'agent_init_euler': agent_init_euler,
            'agent_init_pos_xy_offset': agent_init_pos_xy_offset,
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
        forward(mpm_env, init_state, trajectory.copy(),
                render=args['render_human'],
                use_ray_tracing_renderer=args['use_ray_tracing_renderer'],
                save_loss=args['save_loss'],
                save_gif=args['save_gif'],
                save_img=args['save_img'],
                save_heightmap=args['save_heightmap'],
                img_dir=image_dir_,
                render_init_pcd=args['render_init_pcd'],
                render_end_pcd=args['render_end_pcd'],
                render_heightmap=args['render_heightmap'],
                init_pcd_path=os.path.join(data_path, 'pcd_' + str(data_ind) + str(0) + '.ply'),
                init_pcd_offset=env.target_pcd_offset,
                init_mesh_path=env.mesh_file,
                init_mesh_pos=env.initial_pos)

        print(f'===> CPU memory occupied after forward: {process.memory_percent()} %')
        print(f'===> GPU memory after forward: {get_gpu_memory()}')
        n_episode += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=float, default=4e7, help='Particle density')
    parser.add_argument('--load_params', dest='load_params', default=False, action='store_true', help='Load optimised parameters')
    parser.add_argument('--lp_dataset', dest='load_params_dataset', type=str, default='12mix', help='Dataset used for identifying the loaded optimised parameters')
    parser.add_argument('--lp_run', dest='load_params_run', type=int, default=0, help='Run ID for the optimised parameters')
    parser.add_argument('--lp_seed', dest='load_params_seed', type=int, default=0, help='Seed for the optimised parameters')
    parser.add_argument('--dt', dest='dt', type=float, default=0.01, choices=[0.01, 0.02, 0.03, 0.04], help='Global time step, change this value to verify that the identified parameters are robust against simulation step size.')
    parser.add_argument('--dt_avg', dest='dt_avg', default=False, action='store_true', help='Use dt computed by averaging the MOVEIT! trajectory duration over its time steps')
    parser.add_argument('--con_lv', dest='contact_level', type=int, default=1, choices=[1, 2], help='Examing motions from contact level 1 or 2')
    parser.add_argument('--m_id', dest='motion_ind', type=int, default=1, choices=[1, 2], help='Motion index')
    parser.add_argument('--long_motion', dest='long_motion', default=False, action='store_true', help='Examine long horizon motion simulation. This diseffects the contact_level and motion_ind arguments.')
    parser.add_argument('--agent_id', dest='agent_ind', type=int, default=0, choices=[0, 1, 2], help='Examine the motion executed by which end-effector: 0 - rectangle, 1 - round, 2 - cylinder')
    parser.add_argument('--use_ray_tracing_renderer', dest='use_ray_tracing_renderer', default=False, action='store_true', help='Use the ray tracing renderer for rendering the simulation')
    parser.add_argument('--r_human', dest='render_human', default=False, action='store_true', help='Render the simulation with a pop-up window')
    parser.add_argument('--r_init_pcd', dest='render_init_pcd', default=False, action='store_true', help='Render the initial point cloud with a pop-up window')
    parser.add_argument('--r_end_pcd', dest='render_end_pcd', default=False, action='store_true', help='Render the final point cloud with a pop-up window')
    parser.add_argument('--r_hm', dest='render_heightmap', default=False, action='store_true', help='Render the final height map with a pop-up window')
    parser.add_argument('--save_loss', dest='save_loss', default=False, action='store_true', help='Save the loss information')
    parser.add_argument('--save_img', dest='save_img', default=False, action='store_true', help='Save the images')
    parser.add_argument('--save_heightmap', dest='save_heightmap', default=False, action='store_true', help='Save the height map')
    parser.add_argument('--save_gif', dest='save_gif', default=False, action='store_true', help='Save the simulation as a .gif file')
    parser.add_argument('--img_dir', dest='img_dir', type=str, default=None, help='Directory to save the images, if unspecified, the images will be saved in the default directory')
    args = vars(parser.parse_args())
    main(args)
