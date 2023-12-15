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


def forward_backward(mpm_env, init_state, trajectory, backward=True):
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

    if backward:
        # backward
        mpm_env.reset_grad()
        mpm_env.get_final_loss_grad()
        for i in range(mpm_env.horizon - 1, -1, -1):
            action = trajectory[i]
            mpm_env.step_grad(action=action)

        t3 = time()
        print(f'===> forward: {t2 - t1:.2f}s backward: {t3 - t2:.2f}s')

    return loss_info


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
        'exponential_distance': args['exponential_distance'],
        'averaging_loss': args['averaging_loss'],
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

    grads = []

    for motion_ind in ['1', '2']:
        trajectory = np.load(os.path.join(script_path, '..', f'data-motion-{motion_ind}', 'tr_eef_v.npy'))
        dt_global = np.load(os.path.join(script_path, '..', f'data-motion-{motion_ind}', 'tr_dt.npy'))
        horizon = trajectory.shape[0]
        n_substeps = 50

        for agent in ['rectangle', 'round', 'cylinder']:
            if agent == 'rectangle':
                agent_init_euler = (0, 0, 45)
            else:
                agent_init_euler = (0, 0, 0)
            training_data_path = os.path.join(script_path, '..', f'data-motion-{motion_ind}', f'eef-{agent}')
            data_ids = np.random.randint(9, size=3, dtype=np.int32).tolist()
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
                env, mpm_env, init_state = make_env(data_cfg, env_cfg, loss_cfg)
                print(f'===> Num. simulation particles: {mpm_env.loss.n_particles_matching_mat}')
                print(f'===> Num. target pcd points: {mpm_env.loss.n_target_pcd_points}')
                print(f'===> Num. target particles: {mpm_env.loss.n_target_particles_from_mesh}')
                print(f'===> CPU memory occupied after create env: {process.memory_percent()} %')
                print(f'===> GPU memory after create env: {get_gpu_memory()}')

                for _ in range(5):
                    E = np.asarray(np.random.uniform(E_range[0], E_range[1]), dtype=DTYPE_NP).reshape(
                        (1,))  # Young's modulus
                    nu = np.asarray(np.random.uniform(nu_range[0], nu_range[1]), dtype=DTYPE_NP).reshape(
                        (1,))  # Poisson's ratio
                    yield_stress = np.asarray(np.random.uniform(yield_stress_range[0], yield_stress_range[1]),
                                              dtype=DTYPE_NP).reshape((1,))  # Yield stress

                    set_parameters(mpm_env, env_cfg['material_id'], E, nu, yield_stress)

                    loss_info = forward_backward(mpm_env, init_state, trajectory.copy(), backward=True)

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

                    abort = False
                    if loss_info['point_distance_rs'] < 1e-20:
                        abort = True
                    if (loss_info['point_distance_sr'] > 100) and args['averaging_loss']:
                        abort = True
                    if (loss_info['point_distance_sr'] > 20000) and (not args['averaging_loss']):
                        abort = True
                    if (np.isinf(loss_info['point_distance_sr'])) or (np.isnan(loss_info['point_distance_sr'])):
                        abort = True
                    grad = np.array([mpm_env.simulator.particle_param.grad[material_id].E,
                                     mpm_env.simulator.particle_param.grad[material_id].nu,
                                     mpm_env.simulator.system_param.grad[None].yield_stress,
                                     mpm_env.simulator.particle_param.grad[material_id].rho,
                                     mpm_env.simulator.system_param.grad[None].manipulator_friction,
                                     mpm_env.simulator.system_param.grad[None].ground_friction], dtype=DTYPE_NP)
                    if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
                        abort = True

                    if abort:
                        print(f'===> [Warning] Strange loss or gradient.')
                        print(f'===> [Warning] E: {E}, nu: {nu}, yield stress: {yield_stress}')
                        print(f'===> [Warning] Motion: {motion_ind}, agent: {agent}, data: {data_ind}')
                        abn = {
                            'E': float(E),
                            'nu': float(nu),
                            'yield_stress': float(yield_stress),
                            'motion': motion_ind,
                            'agent': agent,
                            'data': data_ind,
                        }
                        x_np = np.zeros((mpm_env.simualtor.n_particles, 3), dtype=DTYPE_NP)
                        v_np = np.zeros((mpm_env.simualtor.n_particles, 3), dtype=DTYPE_NP)
                        C_np = np.zeros((mpm_env.simualtor.n_particles, 3, 3), dtype=DTYPE_NP)
                        F_np = np.zeros((mpm_env.simualtor.n_particles, 3, 3), dtype=DTYPE_NP)
                        used_np = np.zeros((mpm_env.simualtor.n_particles,), dtype=np.int32)
                        mpm_env.simualtor.readframe(mpm_env.simualtor.cur_substep_local, x_np, v_np, C_np, F_np, used_np)
                        abn['x'] = x_np
                        abn['v'] = v_np
                        abn['C'] = C_np
                        abn['F'] = F_np
                        abn['used'] = used_np
                        abn['actions'] = mpm_env.simualtor.actions_buffer

                        m = 0
                        while True:
                            abnormal_file_name = os.path.join(gradient_file_path, f'abnormal-{str(m)}.json')
                            if not os.path.exists(abnormal_file_name):
                                break
                            m += 1
                        with open(abnormal_file_name, 'w') as f_abn:
                            json.dump(abn, f_abn, indent=2)
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
    print(f"Avg. gradient of manipulator friction: {grad_mean[4]}, std: {grad_std[4]}")
    print(f"Avg. gradient of ground friction: {grad_mean[5]}, std: {grad_std[5]}")
    n = 0
    while True:
        grad_mean_file_name = os.path.join(gradient_file_path, f'grads-mean-{str(n)}.npy')
        grad_std_file_name = os.path.join(gradient_file_path, f'grads-std-{str(n)}.npy')
        if not os.path.exists(grad_mean_file_name):
            break
        n += 1

    np.save(grad_mean_file_name, grad_mean)
    np.save(grad_std_file_name, grad_std)
    with open(os.path.join(gradient_file_path, f'loss-config-{str(n)}.json'), 'w') as f_ac:
        json.dump(loss_cfg, f_ac, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=float, default=4e7)
    parser.add_argument('--dsvs', dest='down_sample_voxel_size', type=float, default=0.005)
    parser.add_argument('--exp_dist', dest='exponential_distance', default=False, action='store_true')
    parser.add_argument('--avg_loss', dest='averaging_loss', default=False, action='store_true')
    parser.add_argument('--pd_rs_loss', dest='point_distance_rs_loss', default=False, action='store_true')
    parser.add_argument('--pd_sr_loss', dest='point_distance_sr_loss', default=False, action='store_true')
    parser.add_argument('--prd_rs_loss', dest='particle_distance_rs_loss', default=False, action='store_true')
    parser.add_argument('--prd_sr_loss', dest='particle_distance_sr_loss', default=False, action='store_true')
    parser.add_argument('--hm_loss', dest='height_map_loss', default=False, action='store_true')
    parser.add_argument('--emd_p_loss', dest='emd_point_distance_loss', default=False, action='store_true')
    parser.add_argument('--emd_pr_loss', dest='emd_particle_distance_loss', default=False, action='store_true')
    args = vars(parser.parse_args())
    main(args)
