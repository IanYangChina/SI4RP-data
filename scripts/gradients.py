import numpy as np
import os
import taichi as ti
from time import time
from doma.engine.utils.misc import get_gpu_memory
import psutil
import json
import argparse
from doma.envs.sys_id_env import make_env, set_parameters

DTYPE_NP = np.float32
DTYPE_TI = ti.f32


def forward_backward(mpm_env, init_state, trajectory, backward=True, debug_grad=False):
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
            if i == (mpm_env.horizon // 2):
                _ = mpm_env.simulator.particle_param.grad[2].E

        t3 = time()
        print(f'===> forward: {t2 - t1:.2f}s backward: {t3 - t2:.2f}s')

    return loss_info


def main(args):
    if args['debug']:
        print('[Warning] Debug mode on, printing gradients.')
    process = psutil.Process(os.getpid())
    script_path = os.path.dirname(os.path.realpath(__file__))
    if args['param_set'] == 0:
        gradient_file_path = os.path.join(script_path, '..', 'gradients-E-nu-ys-rho'+args['dir_suffix'])
    else:
        gradient_file_path = os.path.join(script_path, '..', 'gradients-mf-gf'+args['dir_suffix'])

    os.makedirs(gradient_file_path, exist_ok=True)
    np.random.seed(1)

    material_id = 2

    E_range = (10000, 100000)
    nu_range = (0.001, 0.49)
    yield_stress_range = (50, 2000)

    p_density = args['ptcl_density']
    loss_cfg = {
        'exponential_distance': args['exponential_distance'],
        'averaging_loss': False,
        'point_distance_rs_loss': args['point_distance_rs_loss'],
        'point_distance_sr_loss': args['point_distance_sr_loss'],
        'particle_distance_rs_loss': args['particle_distance_rs_loss'],
        'particle_distance_sr_loss': args['particle_distance_sr_loss'],
        'height_map_loss': args['height_map_loss'],
        'emd_point_distance_loss': args['emd_point_distance_loss'],
        'emd_particle_distance_loss': args['emd_particle_distance_loss'],
        'ptcl_density': p_density,
        'down_sample_voxel_size': args['down_sample_voxel_size'],
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
        if not os.path.exists(loss_cfg_file_name):
            break
        n += 1
    if not args['debug']:
        with open(loss_cfg_file_name, 'w') as f_ac:
            json.dump(loss_cfg, f_ac, indent=2)

    grads = []
    if args['param_set'] == 0:
        motions = ['1', '2']
    else:
        motions = ['3', '4']
    for motion_ind in motions:
        trajectory = np.load(os.path.join(script_path, '..', f'data-motion-{motion_ind}', 'tr_eef_v.npy'))
        dt_global = np.load(os.path.join(script_path, '..', f'data-motion-{motion_ind}', 'tr_dt.npy'))
        horizon = trajectory.shape[0]
        n_substeps = 50

        for agent in [
            'rectangle',
            'round',
            'cylinder'
        ]:
            if agent == 'rectangle':
                agent_init_euler = (0, 0, 45)
            else:
                agent_init_euler = (0, 0, 0)
            training_data_path = os.path.join(script_path, '..', f'data-motion-{motion_ind}', f'eef-{agent}')
            if args['param_set'] == 0:
                data_ids = np.random.randint(9, size=3, dtype=np.int32).tolist()
            else:
                data_ids = np.random.randint(5, size=3, dtype=np.int32).tolist()

            for data_ind in data_ids:
                ti.reset()
                ti.init(arch=ti.opengl, default_fp=DTYPE_TI, default_ip=ti.i32,
                        fast_math=False, random_seed=1)
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
                env, mpm_env, init_state = make_env(data_cfg, env_cfg, loss_cfg, debug_grad=args['debug'])
                print(f'===> Num. simulation particles: {mpm_env.loss.n_particles_matching_mat}')
                print(f'===> Num. target pcd points: {mpm_env.loss.n_target_pcd_points}')
                print(f'===> Num. target particles: {mpm_env.loss.n_target_particles_from_mesh}')
                print(f'===> CPU memory occupied after create env: {process.memory_percent()} %')
                print(f'===> GPU memory after create env: {get_gpu_memory()}')

                for _ in range(10):
                    if args['param_set'] == 0:
                        E = np.asarray(np.random.uniform(E_range[0], E_range[1]), dtype=DTYPE_NP).reshape(
                            (1,))  # Young's modulus
                        nu = np.asarray(np.random.uniform(nu_range[0], nu_range[1]), dtype=DTYPE_NP).reshape(
                            (1,))  # Poisson's ratio
                        yield_stress = np.asarray(np.random.uniform(yield_stress_range[0], yield_stress_range[1]),
                                                  dtype=DTYPE_NP).reshape((1,))  # Yield stress
                        rho = np.asarray(np.random.uniform(100, 2000), dtype=DTYPE_NP).reshape((1,))  # Density
                        ground_friction = np.array([2.0], dtype=DTYPE_NP).reshape((1,))
                        manipulator_friction = np.array([0.0], dtype=DTYPE_NP).reshape((1,))
                    else:
                        E = np.array([40000], dtype=DTYPE_NP).reshape((1,))
                        nu = np.array([0.4], dtype=DTYPE_NP).reshape((1,))
                        yield_stress = np.array([1000], dtype=DTYPE_NP).reshape((1,))
                        rho = np.array([1000], dtype=DTYPE_NP).reshape((1,))
                        ground_friction = np.asarray(np.random.uniform(0.0, 2.0), dtype=DTYPE_NP).reshape((1,))
                        manipulator_friction = np.asarray(np.random.uniform(0.0, 2.0), dtype=DTYPE_NP).reshape((1,))

                    set_parameters(mpm_env, env_cfg['material_id'],
                                   E.copy(), nu.copy(), yield_stress.copy(),
                                   rho.copy(), ground_friction.copy(), manipulator_friction.copy())

                    loss_info = forward_backward(mpm_env, init_state, trajectory.copy(), backward=True, debug_grad=args['debug'])

                    print(f'===> CPU memory occupied after forward-backward: {process.memory_percent()} %')
                    print(f'===> GPU memory after forward-backward: {get_gpu_memory()}')
                    print('===> Gradients:')
                    if args['param_set'] == 0:
                        print(f"Gradient of E: {mpm_env.simulator.particle_param.grad[material_id].E}")
                        print(f"Gradient of nu: {mpm_env.simulator.particle_param.grad[material_id].nu}")
                        print(f"Gradient of rho: {mpm_env.simulator.particle_param.grad[material_id].rho}")
                        print(f"Gradient of yield stress: {mpm_env.simulator.system_param.grad[None].yield_stress}")
                        # print(f"Gradient of theta_c: {mpm_env.simulator.system_param.grad[None].theta_c}")
                        # print(f"Gradient of theta_s: {mpm_env.simulator.system_param.grad[None].theta_s}")
                    else:
                        print(f"Gradient of manipulator friction: {mpm_env.simulator.system_param.grad[None].manipulator_friction}")
                        print(f"Gradient of ground friction: {mpm_env.simulator.system_param.grad[None].ground_friction}")

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
                        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
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
                    else:
                        grads.append(grad.copy())

                mpm_env.simulator.clear_ckpt()

    grad_mean = np.mean(grads, axis=0)
    grad_std = np.std(grads, axis=0)
    print('===> Avg. Gradients:')
    if args['param_set'] == 0:
        print(f"Avg. gradient of E: {grad_mean[0]}, std: {grad_std[0]}")
        print(f"Avg. gradient of nu: {grad_mean[1]}, std: {grad_std[1]}")
        print(f"Avg. gradient of yield stress: {grad_mean[2]}, std: {grad_std[2]}")
        print(f"Avg. gradient of rho: {grad_mean[3]}, std: {grad_std[3]}")
    else:
        print(f"Avg. gradient of manipulator friction: {grad_mean[4]}, std: {grad_std[4]}")
        print(f"Avg. gradient of ground friction: {grad_mean[5]}, std: {grad_std[5]}")

    if not args['debug']:
        np.save(grad_mean_file_name, grad_mean)
        np.save(grad_std_file_name, grad_std)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_suffix', dest='dir_suffix', type=str, default='')
    parser.add_argument('--param_set', dest='param_set', type=int, default=0)
    parser.add_argument('--debug', dest='debug', default=False, action='store_true')
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=float, default=4e7)
    parser.add_argument('--dsvs', dest='down_sample_voxel_size', type=float, default=0.005)
    parser.add_argument('--exp_dist', dest='exponential_distance', default=False, action='store_true')
    parser.add_argument('--pd_rs_loss', dest='point_distance_rs_loss', default=False, action='store_true')
    parser.add_argument('--pd_sr_loss', dest='point_distance_sr_loss', default=False, action='store_true')
    parser.add_argument('--prd_rs_loss', dest='particle_distance_rs_loss', default=False, action='store_true')
    parser.add_argument('--prd_sr_loss', dest='particle_distance_sr_loss', default=False, action='store_true')
    parser.add_argument('--hm_loss', dest='height_map_loss', default=False, action='store_true')
    parser.add_argument('--emd_p_loss', dest='emd_point_distance_loss', default=False, action='store_true')
    parser.add_argument('--emd_pr_loss', dest='emd_particle_distance_loss', default=False, action='store_true')
    args = vars(parser.parse_args())
    main(args)
