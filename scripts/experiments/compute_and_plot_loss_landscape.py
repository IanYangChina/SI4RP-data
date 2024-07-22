import argparse
import logging
from doma.engine.utils.misc import plot_loss_landscape
import numpy as np
import os
import taichi as ti
from doma.envs.sys_id_env import make_env, set_parameters
from time import time

script_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_path, '..')
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
    contact_level = args['contact_level']
    dataset = args['dataset']
    assert dataset in ['12mix', '6mix', '1cyl', '1rec', '1round'], 'dataset must be 12mix, 6mix, 1cyl, 1rec, or 1round'
    motion_name = 'poking' if contact_level == 1 else 'poking-shifting'

    fig_data_path = os.path.join(script_path, '..', 'loss-landscapes-analysis', f'level{contact_level}-{dataset}')
    os.makedirs(fig_data_path, exist_ok=True)

    p_density = args['ptcl_density']
    p_density_str = f'{p_density}pd'
    dt_global = 0.01
    n_substeps = 50

    # we compute all the losses, so doesn't really matter whether they are enabled or not
    loss_cfg = {
        'exponential_distance': False,
        'averaging_loss': False,
        'point_distance_rs_loss': True,
        'point_distance_sr_loss': True,
        'down_sample_voxel_size': 0.005,
        'particle_distance_rs_loss': True,
        'particle_distance_sr_loss': True,
        'voxelise_res': 1080,
        'ptcl_density': p_density,
        'load_height_map': True,
        'height_map_loss': True,
        'height_map_res': 32,
        'height_map_size': 0.11,
        'emd_point_distance_loss': True,
        'emd_particle_distance_loss': True,
    }

    param_pair_id = args['param_pair']
    assert param_pair_id in [0, 1, 2], 'param_pair_id must be 0, 1, or 2'
    if contact_level == 1:
        assert param_pair_id != 2, 'param_pair_id must not be 2 for contact_level 1'

    param_pairs = ['E-nu', 'rho-yieldstress', 'mf-gf']
    xy_param = param_pairs[param_pair_id]
    if param_pair_id == 0:
        p1_list = np.arange(1e4, 3e5, 10000).astype(DTYPE_NP)
        p2_list = np.arange(0.01, 0.49, 0.016).astype(DTYPE_NP)
        E, nu = np.meshgrid(p1_list, p2_list)
        yield_stress = 6e3
        rho = 1300.0
        mf = 0.2
        gf = 2.0
        fig_x_label = 'E'
        fig_y_label = 'nu'
        fig_title_params = f'yield_stress = {yield_stress}, rho = {rho}, \n' + \
                           f'm_friction = {mf}, g_friction = {gf}'
    elif param_pair_id == 1:
        p1_list = np.arange(1000, 1999, 33.3).astype(DTYPE_NP)
        p2_list = np.arange(1000, 20200, 640).astype(DTYPE_NP)
        rho, yield_stress = np.meshgrid(p1_list, p2_list)
        E = 2e5
        nu = 0.2
        mf = 0.2
        gf = 2.0
        fig_x_label = 'rho'
        fig_y_label = 'yield_stress'
        fig_title_params = f'E = {E}, nu = {nu}, \n' + \
                           f'm_friction = {mf}, g_friction = {gf}'
    else:
        p1_list = np.arange(0.05, 2.0, 0.065).astype(DTYPE_NP)
        p2_list = np.arange(0.05, 2.0, 0.065).astype(DTYPE_NP)
        mf, gf = np.meshgrid(p1_list, p2_list)
        E = 200000
        nu = 0.4
        yield_stress = 6e3
        rho = 1300.0
        fig_x_label = 'm_friction'
        fig_y_label = 'g_friction'
        fig_title_params = f'E = {E}, nu = {nu}, \n' + \
                           f'yield_stress = {yield_stress}, rho = {rho}'

    distance_type = 'euclidean'
    if args['read_and_plot']:
        loss_types_to_plot = ['chamfer_loss_pcd',
                              'chamfer_loss_particle',
                              'emd_point_distance_loss',
                              'emd_particle_distance_loss']

        for i in range(len(loss_types_to_plot)):
            loss_type = loss_types_to_plot[i]
            loss = np.load(os.path.join(fig_data_path, f'{loss_type}_{distance_type}_{xy_param}-{p_density_str}.npy'))
            loss -= np.mean(loss)
            plot_fig_title = False
            if plot_fig_title:
                fig_title = f'{loss_type}\n' + fig_title_params
            else:
                fig_title = None
            plot_loss_landscape(E, nu, loss, fig_title=fig_title, colorbar=False, cmap='YlGnBu',
                                x_label=fig_x_label, y_label=fig_y_label,
                                hm=True, show=False, save=True,
                                path=os.path.join(fig_data_path,
                                                  f"{loss_type}_{distance_type}_landscape_{xy_param}-topview-{p_density_str}.pdf"))
        return

    logging.basicConfig(level=logging.NOTSET, filemode="w",
                        filename=os.path.join(fig_data_path,
                                              f'loss_landscape_{distance_type}_{xy_param}_{p_density_str}.log'),
                        format="%(asctime)s %(levelname)s %(message)s")

    point_distance_sr = np.zeros_like(E)
    point_distance_rs = np.zeros_like(E)
    chamfer_loss_pcd = np.zeros_like(E)
    particle_distance_sr = np.zeros_like(E)
    particle_distance_rs = np.zeros_like(E)
    chamfer_loss_particle = np.zeros_like(E)
    height_map_loss_pcd = np.zeros_like(E)
    emd_point_distance_loss = np.zeros_like(E)
    emd_particle_distance_loss = np.zeros_like(E)

    # datasets
    if dataset == '12mix':
        n_datapoints = 12
        motion_ids = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
        agent_ids = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
        data_inds = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    elif dataset == '6mix':
        n_datapoints = 6
        motion_ids = [1, 1, 1, 2, 2, 2]
        agent_ids = [0, 1, 2, 0, 1, 2]
        data_inds = [0, 0, 0, 0, 0, 0]
    else:
        n_datapoints = 1
        motion_ids = [2]
        data_inds = [0]

        if dataset == '1cyl':
            agent_ids = [2]
        elif dataset == '1rec':
            agent_ids = [0]
        else:
            agent_ids = [1]
    # Load trajectories.
    for n in range(n_datapoints):
        motion_id = motion_ids[n]
        agent = agents[agent_ids[n]]
        if agent == 'rectangle':
            agent_init_euler = (0, 0, 45)
        else:
            agent_init_euler = (0, 0, 0)
        trajectory = np.load(os.path.join(script_path, '..', 'trajectories',
                                          f'tr_{motion_name}_{motion_id}_v_dt_{dt_global:0.2f}.npy'))
        horizon = trajectory.shape[0]
        training_data_path = os.path.join(script_path, '..', 'data',
                                          f'data-motion-{motion_name}-{motion_id}', f'eef-{agent}')
        data_id = data_inds[n]

        ti.reset()
        ti.init(arch=backend, default_ip=ti.i32, default_fp=DTYPE_TI, fast_math=True, random_seed=1,
                device_memory_GB=3)
        data_cfg = {
            'data_path': training_data_path,
            'data_ind': str(data_id),
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

        t0 = time()
        for i in range(len(p1_list)):
            for j in range(len(p2_list)):
                if param_pair_id == 0:
                    E = p1_list[i]
                    nu = p2_list[j]
                elif param_pair_id == 1:
                    rho = p1_list[i]
                    yield_stress = p2_list[j]
                else:
                    mf = p1_list[i]
                    gf = p2_list[j]
                set_parameters(mpm_env, env_cfg['material_id'], E, nu,
                               yield_stress=yield_stress, rho=rho,
                               manipulator_friction=mf, ground_friction=gf)
                mpm_env.set_state(init_state['state'], grad_enabled=False)
                for k in range(mpm_env.horizon):
                    action = trajectory[k].copy()
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
                    print(f'===> [Warning] E: {E}, nu: {nu}', f'rho: {rho}, yield_stress: {yield_stress}', f'mf: {mf}, gf: {gf}')
                    print(f'===> [Warning] Motion: {motion_id}, agent: {agent}, data: {data_id}')
                    logging.error(f'===> [Warning] Strange loss.')
                    logging.error(f'===> [Warning] E: {E}, nu: {nu}', f'rho: {rho}, yield_stress: {yield_stress}', f'mf: {mf}, gf: {gf}')
                    logging.error(f'===> [Warning] Motion: {motion_id}, agent: {agent}, data: {data_id}')

                point_distance_sr[j, i] += loss_info['point_distance_sr']
                point_distance_rs[j, i] += loss_info['point_distance_rs']
                chamfer_loss_pcd[j, i] += loss_info['chamfer_loss_pcd']
                particle_distance_sr[j, i] += loss_info['particle_distance_sr']
                particle_distance_rs[j, i] += loss_info['particle_distance_rs']
                chamfer_loss_particle[j, i] += loss_info['chamfer_loss_particle']
                height_map_loss_pcd[j, i] += loss_info['height_map_loss_pcd']
                emd_point_distance_loss[j, i] += loss_info['emd_point_distance_loss']
                emd_particle_distance_loss[j, i] += loss_info['emd_particle_distance_loss']

        print(f'Time taken for data point: {training_data_path}::{data_id}: {time() - t0}')
        logging.info(f'Time taken for data point: {training_data_path}::{data_id}: {time() - t0}')

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

    loss_types = ['point_distance_sr', 'point_distance_rs', 'chamfer_loss_pcd',
                  'particle_distance_sr', 'particle_distance_rs', 'chamfer_loss_particle',
                  'height_map_loss_pcd',
                  'emd_point_distance_loss', 'emd_particle_distance_loss'
                  ]

    for i in range(len(losses)):
        np.save(os.path.join(fig_data_path, f'{loss_types[i]}_{distance_type}_{xy_param}-{p_density_str}.npy'),
                losses[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--con_lv', dest='contact_level', default=0, type=int)
    parser.add_argument('--p_pair', dest='param_pair', default=0, type=int)
    parser.add_argument('--rap', dest='read_and_plot', default=True, action='store_true')
    parser.add_argument('--dataset', dest='dataset', type=str, default='12mix',
                        choices=['12mix', '6mix', '1cyl', '1rec', '1round'])
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=float, default=4e7)
    parser.add_argument('--backend', dest='backend', default='opengl', type=str)
    args = vars(parser.parse_args())
    main(args)
