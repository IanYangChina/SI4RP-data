import numpy as np
import os
import taichi as ti
from time import time
from doma.engine.utils.misc import get_gpu_memory
import psutil
import json
import argparse
from doma.envs.sys_id_env import make_env, set_parameters
import logging

DTYPE_NP = np.float32
DTYPE_TI = ti.f32


def forward_backward(mpm_env, init_state, trajectory, logger, backward=True, debug_grad=False):
    cmap = 'Greys'

    t1 = time()
    mpm_env.set_state(init_state['state'], grad_enabled=True)
    for i in range(mpm_env.horizon):
        action = trajectory[i]
        mpm_env.step(action)

    loss_info = mpm_env.get_final_loss()
    for i, v in loss_info.items():
        if i != 'final_height_map':
            print(f'===> {i}: {v:.4f}')
            logger.info(f'===> {i}: {v:.4f}')
        else:
            pass

    t2 = time()
    # input("===> Press Enter to continue...")

    if backward:
        # backward
        debug_mode = 'l'
        if debug_grad:
            if debug_mode == 'l':
                ans = input("===> Press Enter to proceed one step, or input 'c' to run to the end...")
                if ans == 'c':
                    debug_mode = 'c'

        mpm_env.reset_grad()
        if debug_grad:
            print('******Initial grads:')
            print(f"Gradient of E: {mpm_env.simulator.particle_param.grad[2].E}")
            print(f"Gradient of nu: {mpm_env.simulator.particle_param.grad[2].nu}")
            print(f"Gradient of rho: {mpm_env.simulator.particle_param.grad[2].rho}")
            print(f"Gradient of yield stress: {mpm_env.simulator.system_param.grad[None].yield_stress}")
            print(f"Gradient of manipulator friction: {mpm_env.simulator.system_param.grad[None].manipulator_friction}")
            print(f"Gradient of ground friction: {mpm_env.simulator.system_param.grad[None].ground_friction}")
        mpm_env.get_final_loss_grad()
        if debug_grad:
            print('******Grads after get_final_loss_grad()')
            print(f"Gradient of E: {mpm_env.simulator.particle_param.grad[2].E}")
            print(f"Gradient of nu: {mpm_env.simulator.particle_param.grad[2].nu}")
            print(f"Gradient of rho: {mpm_env.simulator.particle_param.grad[2].rho}")
            print(f"Gradient of yield stress: {mpm_env.simulator.system_param.grad[None].yield_stress}")
            print(f"Gradient of manipulator friction: {mpm_env.simulator.system_param.grad[None].manipulator_friction}")
            print(f"Gradient of ground friction: {mpm_env.simulator.system_param.grad[None].ground_friction}")

        for i in range(mpm_env.horizon - 1, -1, -1):
            action = trajectory[i]
            mpm_env.step_grad(action=action)
            if debug_grad:
                print(f'******Grads after step_grad() at step {i}')
                print(f"Gradient of E: {mpm_env.simulator.particle_param.grad[2].E}")
                print(f"Gradient of nu: {mpm_env.simulator.particle_param.grad[2].nu}")
                print(f"Gradient of rho: {mpm_env.simulator.particle_param.grad[2].rho}")
                print(f"Gradient of yield stress: {mpm_env.simulator.system_param.grad[None].yield_stress}")
                print(f"Gradient of manipulator friction: {mpm_env.simulator.system_param.grad[None].manipulator_friction}")
                print(f"Gradient of ground friction: {mpm_env.simulator.system_param.grad[None].ground_friction}")
                if debug_mode == 'l':
                    ans = input("===> Press Enter to proceed one step, or input 'c' to run to the end...")
                    if ans == 'c':
                        debug_mode = 'c'

            # This is a trick that prevents faulty gradient computation
            # It works for unknown reasons
            _ = mpm_env.simulator.particle_param.grad[2].E

        t3 = time()
        print(f'===> forward: {t2 - t1:.2f}s backward: {t3 - t2:.2f}s')
        logger.info(f'===> forward: {t2 - t1:.2f}s backward: {t3 - t2:.2f}s')

    return loss_info


def main(args):
    if args['backend'] == 'opengl':
        backend = ti.opengl
    elif args['backend'] == 'cuda':
        backend = ti.cuda
    elif args['backend'] == 'vulkan':
        backend = ti.vulkan
    else:
        backend = ti.cpu
    if args['debug']:
        print('[Warning] Debug mode on, printing gradients.')
    process = psutil.Process(os.getpid())
    script_path = os.path.dirname(os.path.realpath(__file__))
    script_path = os.path.join(script_path, '..')
    contact_level = args['contact_level']
    assert contact_level in [1, 2], 'Invalid contact level.'
    gradient_file_path = os.path.join(script_path, '..', 'gradient-analysis',
                                      f'level{contact_level}-12mix')

    os.makedirs(gradient_file_path, exist_ok=True)
    np.random.seed(1)

    material_id = 2
    n_substeps = 50
    dt_global = 0.01

    E_range = (1e4, 3e5)
    nu_range = (0.01, 0.49)
    yield_stress_range = (1000, 2e4)
    rho_range = (1000, 2000)
    f_range = (0.0, 2.0)

    p_density = args['ptcl_density']
    if args['cd_point_distance_loss']:
        point_distance_rs_loss = True
        point_distance_sr_loss = True
    else:
        point_distance_rs_loss = False
        point_distance_sr_loss = False
    if args['cd_particle_distance_loss']:
        particle_distance_rs_loss = True
        particle_distance_sr_loss = True
    else:
        particle_distance_rs_loss = False
        particle_distance_sr_loss = False
    loss_cfg = {
        'exponential_distance': False,
        'averaging_loss': False,
        'point_distance_rs_loss': point_distance_rs_loss,
        'point_distance_sr_loss': point_distance_sr_loss,
        'particle_distance_rs_loss': particle_distance_rs_loss,
        'particle_distance_sr_loss': particle_distance_sr_loss,
        'height_map_loss': args['height_map_loss'],
        'emd_point_distance_loss': args['emd_point_distance_loss'],
        'emd_particle_distance_loss': args['emd_particle_distance_loss'],
        'ptcl_density': p_density,
        'down_sample_voxel_size': 0.005,
        'voxelise_res': 1080,
        'load_height_map': True,
        'height_map_res': 32,
        'height_map_size': 0.11,
    }

    n = 0
    while True:
        grad_mean_file_name = os.path.join(gradient_file_path, f'grads-mean-{str(n)}.npy')
        grad_std_file_name = os.path.join(gradient_file_path, f'grads-std-{str(n)}.npy')
        loss_cfg_file_name = os.path.join(gradient_file_path, f'loss-config-{str(n)}.json')
        log_file_name = os.path.join(gradient_file_path, f'grads_{str(n)}.log')
        if not os.path.exists(loss_cfg_file_name):
            break
        n += 1

    logging.basicConfig(level=logging.NOTSET,filemode="w",
                        filename=log_file_name,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not args['debug']:
        with open(loss_cfg_file_name, 'w') as f_ac:
            json.dump(loss_cfg, f_ac, indent=2)

    grads = []
    motion_name = 'poking' if contact_level == 1 else 'poking-shifting'
    for motion_ind in ['1', '2']:
        trajectory = np.load(os.path.join(script_path, '..', 'data', 'trajectories',
                                          f'tr_{motion_name}_{motion_ind}_v_dt_{dt_global:0.2f}.npy'))
        horizon = trajectory.shape[0]

        for agent in ['rectangle', 'round', 'cylinder']:
            if agent == 'rectangle':
                agent_init_euler = (0, 0, 45)
            else:
                agent_init_euler = (0, 0, 0)
            training_data_path = os.path.join(script_path, '..', 'data',
                                              f'data-motion-{motion_name}-{motion_ind}', f'eef-{agent}')

            for data_ind in [0, 1]:
                ti.reset()
                ti.init(arch=backend, default_fp=DTYPE_TI, default_ip=ti.i32, device_memory_GB=3,
                        debug=False, advanced_optimization=True, fast_math=True,
                        print_ir=False,
                        # log_level=ti.TRACE,
                        # offline_cache=False,
                        random_seed=1)
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
                logging.info(f'===> CPU memory occupied before create env: {process.memory_percent()} %')
                logging.info(f'===> GPU memory before create env: {get_gpu_memory()}')
                env, mpm_env, init_state = make_env(data_cfg, env_cfg, loss_cfg, debug_grad=args['debug'])
                print(f'===> Num. simulation particles: {mpm_env.loss.n_particles_matching_mat}')
                print(f'===> Num. target pcd points: {mpm_env.loss.n_target_pcd_points}')
                print(f'===> Num. target particles: {mpm_env.loss.n_target_particles_from_mesh}')
                print(f'===> CPU memory occupied after create env: {process.memory_percent()} %')
                print(f'===> GPU memory after create env: {get_gpu_memory()}')
                logging.info(f'===> Num. simulation particles: {mpm_env.loss.n_particles_matching_mat}')
                logging.info(f'===> Num. target pcd points: {mpm_env.loss.n_target_pcd_points}')
                logging.info(f'===> Num. target particles: {mpm_env.loss.n_target_particles_from_mesh}')
                logging.info(f'===> CPU memory occupied after create env: {process.memory_percent()} %')
                logging.info(f'===> GPU memory after create env: {get_gpu_memory()}')

                for _ in range(20):
                    E = np.asarray(np.random.uniform(E_range[0], E_range[1]), dtype=DTYPE_NP).reshape(
                        (1,))  # Young's modulus
                    nu = np.asarray(np.random.uniform(nu_range[0], nu_range[1]), dtype=DTYPE_NP).reshape(
                        (1,))  # Poisson's ratio
                    yield_stress = np.asarray(np.random.uniform(yield_stress_range[0], yield_stress_range[1]),
                                              dtype=DTYPE_NP).reshape((1,))  # Yield stress
                    rho = np.asarray(np.random.uniform(rho_range[0], rho_range[1]), dtype=DTYPE_NP).reshape(
                        (1,))  # Density
                    ground_friction = np.array([2.0], dtype=DTYPE_NP).reshape((1,))
                    manipulator_friction = np.array([0.0], dtype=DTYPE_NP).reshape((1,))
                    if args['param_set'] == 1:
                        ground_friction = np.asarray(np.random.uniform(f_range[0], f_range[1]), dtype=DTYPE_NP).reshape((1,))
                        manipulator_friction = np.asarray(np.random.uniform(f_range[0], f_range[1]), dtype=DTYPE_NP).reshape((1,))

                    set_parameters(mpm_env, env_cfg['material_id'],
                                   E.copy(), nu.copy(), yield_stress.copy(),
                                   rho.copy(), ground_friction.copy(), manipulator_friction.copy())

                    loss_info = forward_backward(mpm_env, init_state, trajectory.copy(), logger=logging,
                                                 backward=True, debug_grad=args['debug'])

                    print(f'===> CPU memory occupied after forward-backward: {process.memory_percent()} %')
                    print(f'===> GPU memory after forward-backward: {get_gpu_memory()}')
                    print('===> Gradients:')
                    print(f"Gradient of E: {mpm_env.simulator.particle_param.grad[material_id].E}")
                    print(f"Gradient of nu: {mpm_env.simulator.particle_param.grad[material_id].nu}")
                    print(f"Gradient of rho: {mpm_env.simulator.particle_param.grad[material_id].rho}")
                    print(f"Gradient of yield stress: {mpm_env.simulator.system_param.grad[None].yield_stress}")
                    logging.info(f'===> CPU memory occupied after forward-backward: {process.memory_percent()} %')
                    logging.info(f'===> GPU memory after forward-backward: {get_gpu_memory()}')
                    logging.info('===> Gradients:')
                    logging.info(f"Gradient of E: {mpm_env.simulator.particle_param.grad[material_id].E}")
                    logging.info(f"Gradient of nu: {mpm_env.simulator.particle_param.grad[material_id].nu}")
                    logging.info(f"Gradient of rho: {mpm_env.simulator.particle_param.grad[material_id].rho}")
                    logging.info(f"Gradient of yield stress: {mpm_env.simulator.system_param.grad[None].yield_stress}")
                    if args['param_set'] == 1:
                        print(f"Gradient of manipulator friction: {mpm_env.simulator.system_param.grad[None].manipulator_friction}")
                        print(f"Gradient of ground friction: {mpm_env.simulator.system_param.grad[None].ground_friction}")
                        logging.info(f"Gradient of manipulator friction: {mpm_env.simulator.system_param.grad[None].manipulator_friction}")
                        logging.info(f"Gradient of ground friction: {mpm_env.simulator.system_param.grad[None].ground_friction}")

                    grad = np.array([mpm_env.simulator.particle_param.grad[material_id].E,
                                     mpm_env.simulator.particle_param.grad[material_id].nu,
                                     mpm_env.simulator.system_param.grad[None].yield_stress,
                                     mpm_env.simulator.particle_param.grad[material_id].rho,
                                     mpm_env.simulator.system_param.grad[None].manipulator_friction,
                                     mpm_env.simulator.system_param.grad[None].ground_friction], dtype=DTYPE_NP)

                    # Check if the loss is strange
                    abort = False
                    for i, v in loss_info.items():
                        if i != 'final_height_map':
                            if np.isinf(v) or np.isnan(v):
                                abort = True
                                break
                    if not abort:
                        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)) or np.any(np.abs(grad) > 1e6):
                            abort = True

                    if not abort:
                        num_zero_grad = 0
                        for n in range(6):
                            if grad[n] == 0.0:
                                num_zero_grad += 1
                        if num_zero_grad > 4:
                           abort = True

                    if abort:
                        print(f'===> [Warning] Strange loss or gradient.')
                        print(f'===> [Warning] E: {E}, nu: {nu}, yield stress: {yield_stress}')
                        print(f'===> [Warning] Rho: {rho}, ground friction: {ground_friction}, manipulator friction: {manipulator_friction}')
                        print(f'===> [Warning] Motion: {motion_ind}, agent: {agent}, data: {data_ind}')
                        logging.error(f'===> [Warning] Strange loss or gradient.')
                        logging.error(f'===> [Warning] E: {E}, nu: {nu}, yield stress: {yield_stress}')
                        logging.error(f'===> [Warning] Rho: {rho}, ground friction: {ground_friction}, manipulator friction: {manipulator_friction}')
                        logging.error(f'===> [Warning] Motion: {motion_ind}, agent: {agent}, data: {data_ind}')
                    else:
                        grads.append(grad.copy())

                mpm_env.simulator.clear_ckpt()

    grad_mean = np.mean(grads, axis=0)
    grad_std = np.std(grads, axis=0)
    print('===> Avg. Gradients:')
    print(f"Avg. gradient of E: {grad_mean[0]}, std: {grad_std[0]}")
    print(f"Avg. gradient of nu: {grad_mean[1]}, std: {grad_std[1]}")
    print(f"Avg. gradient of yield stress: {grad_mean[2]}, std: {grad_std[2]}")
    print(f"Avg. gradient of rho: {grad_mean[3]}, std: {grad_std[3]}")
    logging.info('===> Avg. Gradients:')
    logging.info(f"Avg. gradient of E: {grad_mean[0]}, std: {grad_std[0]}")
    logging.info(f"Avg. gradient of nu: {grad_mean[1]}, std: {grad_std[1]}")
    logging.info(f"Avg. gradient of yield stress: {grad_mean[2]}, std: {grad_std[2]}")
    logging.info(f"Avg. gradient of rho: {grad_mean[3]}, std: {grad_std[3]}")
    if contact_level == 2:
        print(f"Avg. gradient of manipulator friction: {grad_mean[4]}, std: {grad_std[4]}")
        print(f"Avg. gradient of ground friction: {grad_mean[5]}, std: {grad_std[5]}")
        logging.info(f"Avg. gradient of manipulator friction: {grad_mean[4]}, std: {grad_std[4]}")
        logging.info(f"Avg. gradient of ground friction: {grad_mean[5]}, std: {grad_std[5]}")

    if not args['debug']:
        np.save(grad_mean_file_name, grad_mean)
        np.save(grad_std_file_name, grad_std)


if __name__ == '__main__':
    description = 'Compute the means and standard deviations of the gradients for material parameters using the 12mix dataset with randomly sampled parameter values.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--con_lv', dest='contact_level', type=int, default=1, help='Contact level: 1 or 2')
    parser.add_argument('--debug', dest='debug', default=False, action='store_true', help='Debug mode, print gradients for every computation kernel.')
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=float, default=4e7, help='Particle density')
    parser.add_argument('--cd_p_loss', dest='cd_point_distance_loss', default=False, action='store_true', help='Count Chamfer loss between real points and simulated particles into loss computation.')
    parser.add_argument('--cd_pr_loss', dest='cd_particle_distance_loss', default=False, action='store_true', help='Count Chamfer loss between reconstructed particles and simulated particles into loss computation.')
    parser.add_argument('--emd_p_loss', dest='emd_point_distance_loss', default=False, action='store_true', help='Count EMD loss from real points to simulated particles into loss computation.')
    parser.add_argument('--emd_pr_loss', dest='emd_particle_distance_loss', default=False, action='store_true', help='Count EMD loss from reconstructed particles to simulated particles into loss computation.')
    parser.add_argument('--hm_loss', dest='height_map_loss', default=False, action='store_true', help='Count height map loss into loss computation.')
    parser.add_argument('--backend', dest='backend', default='cuda', type=str, help='Computation backend: cuda, opengl, or cpu')
    args = vars(parser.parse_args())
    main(args)
