import argparse
from doma.engine.utils.misc import plot_loss_landscape
import numpy as np
import os
import taichi as ti
from doma.envs.sys_id_env import make_env, set_parameters
from time import time
import logging
script_path = os.path.dirname(os.path.realpath(__file__))
DTYPE_NP = np.float32
DTYPE_TI = ti.f32


def main(args):
    if args['backend'] == 'opengl':
        backend = ti.opengl
    elif args['backend'] == 'cuda':
        backend = ti.cuda
    elif args['backend'] == 'vulkan':
        backend = ti.vulkan
    else:
        backend = ti.cpu

    agents = ['rectangle', 'round', 'cylinder']
    if args['param_set'] == 0:
        motion_inds = ['1', '2']
        if args['fewshot']:
            n_datapoints = 12
            fig_data_path = os.path.join(script_path, '..', 'loss-landscapes-m12-few-shot')
        elif args['oneshot']:
            n_datapoints = 6
            fig_data_path = os.path.join(script_path, '..', 'loss-landscapes-m12-one-shot')
        elif args['realoneshot']:
            motion_inds = ['2']
            realoneshot_agent = agents[args['realoneshot_agent_ind']]
            n_datapoints = 1
            fig_data_path = os.path.join(script_path, '..', f'loss-landscapes-m2-realoneshot-{realoneshot_agent}')
        else:
            n_datapoints = 18
            fig_data_path = os.path.join(script_path, '..', 'loss-landscapes-m12')
    else:
        motion_inds = ['3', '4']
        if args['fewshot']:
            n_datapoints = 12
            fig_data_path = os.path.join(script_path, '..', 'loss-landscapes-m34-few-shot')
        elif args['oneshot']:
            n_datapoints = 6
            fig_data_path = os.path.join(script_path, '..', 'loss-landscapes-m34-one-shot')
        elif args['realoneshot']:
            motion_inds = ['4']
            realoneshot_agent = agents[args['realoneshot_agent_ind']]
            n_datapoints = 1
            fig_data_path = os.path.join(script_path, '..', f'loss-landscapes-m4-realoneshot-{realoneshot_agent}')
        else:
            n_datapoints = 18
            fig_data_path = os.path.join(script_path, '..', 'loss-landscapes-m34')

    os.makedirs(fig_data_path, exist_ok=True)

    p_density = args['ptcl_density']
    p_density_str = f'{p_density}pd'

    loss_cfg = {
        'exponential_distance': args['exponential_distance'],
        'averaging_loss': False,
        'point_distance_rs_loss': True,
        'point_distance_sr_loss': False,
        'down_sample_voxel_size': args['down_sample_voxel_size'],
        'particle_distance_rs_loss': False,
        'particle_distance_sr_loss': True,
        'voxelise_res': 1080,
        'ptcl_density': p_density,
        'load_height_map': True,
        'height_map_loss': True,
        'height_map_res': 32,
        'height_map_size': 0.11,
        'emd_point_distance_loss': True,
        'emd_particle_distance_loss': False,
    }

    xy_param = 'rho-yieldstress'
    rho_list = np.arange(1000, 1999, 33.3).astype(DTYPE_NP)
    yield_stress_list = np.arange(1000, 20200, 640).astype(DTYPE_NP)

    rho, yield_stress = np.meshgrid(rho_list, yield_stress_list)
    E = 2e5
    nu = 0.2
    mf = 0.2
    gf = 2.0

    if args['exponential_distance']:
        distance_type = 'exponential'
    else:
        distance_type = 'euclidean'

    if args['read_and_plot']:
        loss_types = ['chamfer_loss_pcd',
                      'chamfer_loss_particle',
                      'emd_point_distance_loss',
                      'emd_particle_distance_loss']

        for i in range(len(loss_types)):
            loss_type = loss_types[i]
            if args['param_set'] == 0:
                if loss_type == 'chamfer_loss_pcd':
                    min_val = 5600
                    max_val = 10500
                elif loss_type == 'chamfer_loss_particle':
                    min_val = 3500
                    max_val = 6800
                elif loss_type == 'emd_point_distance_loss':
                    min_val = 890
                    max_val = 1100
                elif loss_type == 'emd_particle_distance_loss':
                    min_val = 4500
                    max_val = 7200
                else:
                    raise ValueError('Invalid loss type.')
            else:
                if loss_type == 'chamfer_loss_pcd':
                    min_val = 12000
                    max_val = 32000
                elif loss_type == 'chamfer_loss_particle':
                    min_val = 7500
                    max_val = 43000
                elif loss_type == 'emd_point_distance_loss':
                    min_val = 1430
                    max_val = 8100
                elif loss_type == 'emd_particle_distance_loss':
                    min_val = 9000
                    max_val = 78000
                else:
                    raise ValueError('Invalid loss type.')
            loss = np.load(os.path.join(fig_data_path, f'{loss_type}_{distance_type}_{xy_param}-{p_density_str}.npy'))
            plot_loss_landscape(rho, yield_stress, loss, fig_title=None, colorbar=True, cmap='YlGnBu', min_val=min_val, max_val=max_val,
                                x_label='rho', y_label='yield_stress', hm=True, show=False, save=True,
                                path=os.path.join(fig_data_path, f"{loss_type}_{distance_type}_landscape_{xy_param}-topview-{p_density_str}.pdf"))
        return

    logging.basicConfig(level=logging.NOTSET,filemode="w",
                        filename=os.path.join(fig_data_path, f'loss_landscape_{distance_type}_{xy_param}_{p_density_str}.log'),
                        format="%(asctime)s %(levelname)s %(message)s")

    point_distance_sr = np.zeros_like(rho)
    point_distance_rs = np.zeros_like(rho)
    chamfer_loss_pcd = np.zeros_like(rho)
    particle_distance_sr = np.zeros_like(rho)
    particle_distance_rs = np.zeros_like(rho)
    chamfer_loss_particle = np.zeros_like(rho)
    height_map_loss_pcd = np.zeros_like(rho)
    emd_point_distance_loss = np.zeros_like(rho)
    emd_particle_distance_loss = np.zeros_like(rho)

    fewshot_data_id_dict = {
        '1': {'rectangle': [3, 5], 'round': [0, 1], 'cylinder': [1, 2]},
        '2': {'rectangle': [1, 3], 'round': [0, 2], 'cylinder': [0, 2]},
        '3': {'rectangle': [1, 2], 'round': [0, 1], 'cylinder': [0, 4]},
        '4': {'rectangle': [1, 3], 'round': [1, 4], 'cylinder': [0, 4]},
    }
    oneshot_data_id_dict = {
        '1': {'rectangle': [3], 'round': [0], 'cylinder': [1]},
        '2': {'rectangle': [1], 'round': [0], 'cylinder': [0]},
        '3': {'rectangle': [1], 'round': [0], 'cylinder': [0]},
        '4': {'rectangle': [1], 'round': [1], 'cylinder': [0]},
    }
    # Load trajectories.
    for motion_ind in motion_inds:
        dt_global = 0.01
        trajectory = np.load(os.path.join(script_path, '..', 'trajectories', f'tr_{motion_ind}_v_dt_{dt_global:0.2f}.npy'))
        horizon = trajectory.shape[0]
        n_substeps = 50

        if args['realoneshot']:
            agents_to_use = [agents[args['realoneshot_agent_ind']]]
        else:
            agents_to_use = agents
        for agent in agents_to_use:
            training_data_path = os.path.join(script_path, '..', f'data-motion-{motion_ind}', f'eef-{agent}')
            if agent == 'rectangle':
                agent_init_euler = (0, 0, 45)
            else:
                agent_init_euler = (0, 0, 0)
            if args['fewshot']:
                data_ids = fewshot_data_id_dict[motion_ind][agent]
            elif args['oneshot']:
                data_ids = oneshot_data_id_dict[motion_ind][agent]
            elif args['realoneshot']:
                data_ids = oneshot_data_id_dict[motion_ind][agent]
            else:
                data_ids = np.random.choice(5, size=3, replace=False).tolist()
            for data_ind in data_ids:
                ti.reset()
                ti.init(arch=backend, default_ip=ti.i32, default_fp=DTYPE_TI,
                        fast_math=True, random_seed=1,
                        device_memory_GB=3)
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

                env, mpm_env, init_state = make_env(data_cfg, env_cfg, loss_cfg)
                print(f'===> Num. simulation particles: {mpm_env.loss.n_particles_matching_mat}')
                print(f'===> Num. target pcd points: {mpm_env.loss.n_target_pcd_points}')
                print(f'===> Num. target particles: {mpm_env.loss.n_target_particles_from_mesh}')
                print(f'Start calculating losses with grid size: {point_distance_sr.shape}')
                logging.info(f'===> Num. simulation particles: {mpm_env.loss.n_particles_matching_mat}')
                logging.info(f'===> Num. target pcd points: {mpm_env.loss.n_target_pcd_points}')
                logging.info(f'===> Num. target particles: {mpm_env.loss.n_target_particles_from_mesh}')
                logging.info(f'Start calculating losses with grid size: {point_distance_sr.shape}')

                # discard the first compute after env creation
                set_parameters(mpm_env, env_cfg['material_id'],  E, nu,
                               yield_stress=yield_stress_list[0], rho=rho_list[0],
                               manipulator_friction=mf, ground_friction=gf)
                mpm_env.set_state(init_state['state'], grad_enabled=False)
                for k in range(mpm_env.horizon):
                    action = trajectory[k].copy()
                    mpm_env.step(action)
                loss_info = mpm_env.get_final_loss()

                t0 = time()
                for i in range(len(rho_list)):
                    for j in range(len(yield_stress_list)):
                        set_parameters(mpm_env, env_cfg['material_id'], E, nu,
                                       yield_stress_list[j], rho_list[i],
                                       manipulator_friction=0.2, ground_friction=2.0)
                        mpm_env.set_state(init_state['state'], grad_enabled=False)
                        for k in range(mpm_env.horizon):
                            action = trajectory[k]
                            mpm_env.step(action)
                        loss_info = mpm_env.get_final_loss()

                        abort = False
                        print(f'The {i}, {j}-th loss is:')
                        logging.info(f'The {i}, {j}-th loss is:')
                        for b, v in loss_info.items():
                            if b == 'final_height_map':
                                continue
                            # Check if the loss is strange
                            if np.isinf(v) or np.isnan(v):
                                abort = True
                            print(f'{b}: {v:.4f}')
                            logging.info(f'{b}: {v:.4f}')

                        if abort:
                            print(f'===> [Warning] Strange loss.')
                            print(f'===> [Warning] Rho: {rho_list[i]}, ys: {yield_stress_list[j]}')
                            print(f'===> [Warning] Motion: {motion_ind}, agent: {agent}, data: {data_ind}')
                            logging.error(f'===> [Warning] Strange loss.')
                            logging.error(f'===> [Warning] Rho: {rho_list[i]}, ys: {yield_stress_list[j]}')
                            logging.error(f'===> [Warning] Motion: {motion_ind}, agent: {agent}, data: {data_ind}')

                        point_distance_sr[j, i] += loss_info['point_distance_sr']
                        point_distance_rs[j, i] += loss_info['point_distance_rs']
                        chamfer_loss_pcd[j, i] += loss_info['chamfer_loss_pcd']
                        particle_distance_sr[j, i] += loss_info['particle_distance_sr']
                        particle_distance_rs[j, i] += loss_info['particle_distance_rs']
                        chamfer_loss_particle[j, i] += loss_info['chamfer_loss_particle']
                        height_map_loss_pcd[j, i] += loss_info['height_map_loss_pcd']
                        emd_point_distance_loss[j, i] += loss_info['emd_point_distance_loss']
                        emd_particle_distance_loss[j, i] += loss_info['emd_particle_distance_loss']

                # Unnecessary to clear ckpt without gradient enabled.
                # mpm_env.simulator.clear_ckpt()
                print(f'Time taken for data point {data_ind}: {time() - t0}')
                logging.info(f'Time taken for data point {data_ind}: {time() - t0}')

    losses = [point_distance_sr / n_datapoints,
              point_distance_rs / n_datapoints,
              chamfer_loss_pcd / n_datapoints,
              particle_distance_sr / n_datapoints,
              particle_distance_rs / n_datapoints,
              chamfer_loss_particle / n_datapoints,
              height_map_loss_pcd / n_datapoints,
              emd_point_distance_loss / n_datapoints,
              emd_particle_distance_loss / n_datapoints
              ]

    loss_types = [
        'point_distance_sr', 'point_distance_rs', 'chamfer_loss_pcd',
        'particle_distance_sr', 'particle_distance_rs', 'chamfer_loss_particle',
        'height_map_loss_pcd',
        'emd_point_distance_loss', 'emd_particle_distance_loss'
    ]

    for i in range(len(losses)):
        np.save(os.path.join(fig_data_path, f'{loss_types[i]}_{distance_type}_{xy_param}-{p_density_str}.npy'), losses[i])
        fig_title = (f'{loss_types[i]}\n'
                     f'E = {E}, nu = {nu}\n'
                     f'm_friction = {mf}, g_friction = {gf}')
        # plot_loss_landscape(rho, yield_stress, losses[i], fig_title=fig_title, view='left',
        #                     x_label='rho', y_label='yield_stress', z_label='Loss', hm=False, show=False, save=True,
        #                     path=os.path.join(fig_data_path, f"{loss_types[i]}_{distance_type}_landscape_{xy_param}-leftview-{p_density_str}.pdf"))
        # plot_loss_landscape(rho, yield_stress, losses[i], fig_title=fig_title, view='right',
        #                     x_label='rho', y_label='yield_stress', z_label='Loss', hm=False, show=False, save=True,
        #                     path=os.path.join(fig_data_path, f"{loss_types[i]}_{distance_type}_landscape_{xy_param}-rightview-{p_density_str}.pdf"))
        plot_loss_landscape(rho, yield_stress, losses[i], fig_title=None, colorbar=True, cmap='YlGnBu',
                            x_label='rho', y_label='yield_stress', z_label='Loss', hm=True, show=False, save=True,
                            path=os.path.join(fig_data_path, f"{loss_types[i]}_{distance_type}_landscape_{xy_param}-topview-{p_density_str}.pdf"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_set', dest='param_set', default=0, type=int)
    parser.add_argument('--rap', dest='read_and_plot', default=False, action='store_true')
    parser.add_argument('--fewshot', dest='fewshot', default=False, action='store_true')
    parser.add_argument('--oneshot', dest='oneshot', default=False, action='store_true')
    parser.add_argument('--realoneshot', dest='realoneshot', default=False, action='store_true')
    parser.add_argument('--realoneshot_agent_ind', dest='realoneshot_agent_ind', type=int, default=0)
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=float, default=4e7)
    parser.add_argument('--dsvs', dest='down_sample_voxel_size', type=float, default=0.005)
    parser.add_argument('--exp_dist', dest='exponential_distance', default=False, action='store_true')
    parser.add_argument('--backend', dest='backend', default='cuda', type=str)
    args = vars(parser.parse_args())
    main(args)
