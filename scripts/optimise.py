import os
import argparse
import taichi as ti
import numpy as np
from time import time
from torch.utils.tensorboard import SummaryWriter
from doma.optimiser.adam import Adam, SGD
from doma.envs import SysIDEnv
import json

MATERIAL_ID = 2


def forward_backward(mpm_env, init_state, trajectory, render, backward=True):
    # Forward
    t1 = time()
    mpm_env.set_state(init_state['state'], grad_enabled=True)
    for i in range(mpm_env.horizon):
        action = trajectory[i]
        mpm_env.step(action)
        if render:
            mpm_env.render(mode='human')
    loss_info = mpm_env.get_final_loss()

    t2 = time()

    if backward:
        # backward
        mpm_env.reset_grad()
        mpm_env.get_final_loss_grad()
        for i in range(mpm_env.horizon - 1, -1, -1):
            action = trajectory[i]
            mpm_env.step_grad(action=action)

        t3 = time()
        # print(f'=====> forward: {t2 - t1:.2f}s backward: {t3 - t2:.2f}s')

    return loss_info


def set_parameters(mpm_env, E, nu, yield_stress):
    mpm_env.simulator.system_param[None].yield_stress = yield_stress
    mpm_env.simulator.particle_param[MATERIAL_ID].rho = 1000
    mpm_env.simulator.particle_param[MATERIAL_ID].E = E
    mpm_env.simulator.particle_param[MATERIAL_ID].nu = nu


def make_env(data_path, data_ind, horizon,
             ptcl_density, dtype_np,
             agent_name, agent_init_euler, loss_config):
    obj_start_mesh_file_path = os.path.join(data_path, 'mesh_' + data_ind + str(0) + '_repaired_normalised.obj')
    if not os.path.exists(obj_start_mesh_file_path):
        return None, None
    obj_start_centre_real = np.load(
        os.path.join(data_path, 'mesh_' + data_ind + str(0) + '_repaired_centre.npy')).astype(dtype_np)
    obj_start_centre_top_normalised = np.load(
        os.path.join(data_path, 'mesh_' + data_ind + str(0) + '_repaired_normalised_centre_top.npy')).astype(
        dtype_np)

    obj_end_pcd_file_path = os.path.join(data_path, 'pcd_' + data_ind + str(1) + '.ply')
    obj_end_mesh_file_path = os.path.join(data_path, 'mesh_' + data_ind + str(1) + '_repaired_normalised.obj')
    obj_end_centre_top_normalised = np.load(
        os.path.join(data_path, 'mesh_' + data_ind + str(1) + '_repaired_normalised_centre_top.npy')).astype(
        dtype_np)

    # Building environment
    obj_start_initial_pos = np.array([0.25, 0.25, obj_start_centre_top_normalised[-1] + 0.01], dtype=dtype_np)
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

    env = SysIDEnv(ptcl_density=ptcl_density, horizon=horizon, material_id=MATERIAL_ID, voxelise_res=1080,
                   mesh_file=obj_start_mesh_file_path, initial_pos=obj_start_initial_pos,
                   loss_cfg=loss_config,
                   agent_cfg_file=agent_name + '_eef.yaml',
                   agent_init_pos=agent_init_pos,
                   agent_init_euler=agent_init_euler)
    env.reset()
    mpm_env = env.mpm_env
    init_state = mpm_env.get_state()

    return env, mpm_env, init_state


def main(arguments):
    script_path = os.path.dirname(os.path.realpath(__file__))
    DTYPE_NP = np.float32
    DTYPE_TI = ti.f32
    particle_density = 4e8
    assert arguments['hm_res'] in [32, 64], 'height map resolution must be 32 or 64'
    loss_cfg = {
        'point_distance_rs_loss': arguments['pd_rs_loss'],
        'point_distance_sr_loss': arguments['pd_sr_loss'],
        'down_sample_voxel_size': 0.0015,
        'particle_distance_rs_loss': arguments['prd_rs_loss'],
        'particle_distance_sr_loss': arguments['prd_sr_loss'],
        'voxelise_res': 1080,
        'ptcl_density': particle_density,
        'load_height_map': True,
        'height_map_loss': arguments['hm_loss'],
        'height_map_res': arguments['hm_res'],
        'height_map_size': 0.11,
    }
    # Setting up horizon and trajectory
    dt = 0.001
    # Trajectory 1 press down 0.015 m and lifts for 0.03 m
    # In simulation we only simulate the pressing down part
    real_horizon_1 = int(0.03 / dt)
    v = 0.045 / 0.03  # 1.5 m/s
    horizon_1_up = int((0.015 / v) / dt)  # 0.01 s
    horizon_1_down = int((0.03 / v) / dt)  # 0.02 s
    horizon_1 = horizon_1_up + horizon_1_down  # 30 steps
    trajectory_1 = np.zeros(shape=(horizon_1, 6), dtype=DTYPE_NP)
    trajectory_1[:horizon_1_up, 2] = -v
    trajectory_1[horizon_1_up:, 2] = v

    # Trajectory 2 press down 0.02 m and lifts for 0.03 m
    # In simulation we only simulate the pressing down part
    real_horizon_2 = int(0.04 / dt)
    v = 0.05 / 0.045  # 1.11111111 m/s
    horizon_2_up = int((0.02 / v) / dt)  # 0.018 s
    horizon_2_down = int((0.03 / v) / dt)  # 0.027 s
    horizon_2 = horizon_2_up + horizon_2_down  # 45 steps
    trajectory_2 = np.zeros(shape=(horizon_2, 6), dtype=DTYPE_NP)
    trajectory_2[:horizon_2_up, 2] = -v
    trajectory_2[horizon_2_up:, 2] = v

    # Parameter ranges
    E_range = (10000, 100000)
    nu_range = (0.001, 0.49)
    yield_stress_range = (50, 2000)

    n_epoch = 100
    seeds = [0, 1, 2]
    print(f"=====> Optimising for {n_epoch} epochs for {len(seeds)} random seeds.")
    n = 0
    while True:
        log_p_dir = os.path.join(script_path, '..', f'optimisation-run{n}-logs')
        if os.path.exists(log_p_dir):
            n += 1
        else:
            break
    os.makedirs(log_p_dir, exist_ok=True)
    with open(os.path.join(log_p_dir, 'loss_config.json'), 'w') as f_ac:
        json.dump(loss_cfg, f_ac)

    for seed in seeds:
        # Setting up random seed
        np.random.seed(seed)
        # Logger
        log_dir = os.path.join(log_p_dir, f'seed-{seed}')
        os.makedirs(log_dir, exist_ok=True)
        logger = SummaryWriter(log_dir=log_dir)

        # Initialising parameters
        E = np.asarray(np.random.uniform(E_range[0], E_range[1]), dtype=DTYPE_NP).reshape((1,))  # Young's modulus
        nu = np.asarray(np.random.uniform(nu_range[0], nu_range[1]), dtype=DTYPE_NP).reshape((1,))  # Poisson's ratio
        yield_stress = np.asarray(np.random.uniform(yield_stress_range[0], yield_stress_range[1]),
                                  dtype=DTYPE_NP).reshape((1,))  # Yield stress

        print(f"=====> Seed: {seed}, initial parameters: E={E}, nu={nu}, yield_stress={yield_stress}")
        if arguments['adam']:
            # Optimiser: Adam
            optim_E = Adam(parameters_shape=E.shape,
                           cfg={'lr': 1e7, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
            optim_nu = Adam(parameters_shape=nu.shape,
                            cfg={'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
            optim_yield_stress = Adam(parameters_shape=yield_stress.shape,
                                      cfg={'lr': 1e4, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
        else:
            # Optimiser: SGD
            optim_E = SGD(parameters_shape=E.shape,
                          cfg={'lr': 1e7})
            optim_nu = SGD(parameters_shape=nu.shape,
                           cfg={'lr': 0.001})
            optim_yield_stress = SGD(parameters_shape=yield_stress.shape,
                                     cfg={'lr': 1e4})

        motion_inds = ['1', '2']
        agents = ['rectangle', 'round', 'cylinder']
        data_inds = np.arange(9).astype(int)
        num_datapoints = len(data_inds) * len(agents) * len(motion_inds)

        for epoch in range(n_epoch):
            t1 = time()
            loss = {
                'avg_point_distance_sr': 0.0,
                'avg_point_distance_rs': 0.0,
                'chamfer_loss_pcd': 0.0,
                'avg_particle_distance_sr': 0.0,
                'avg_particle_distance_rs': 0.0,
                'chamfer_loss_particle': 0.0,
                'height_map_loss_pcd': 0.0,
                'total_loss': 0.0
            }
            grads = np.zeros(shape=(3,), dtype=DTYPE_NP)
            for motion_ind in motion_inds:
                if motion_ind == '1':
                    horizon = horizon_1
                    trajectory = trajectory_1
                else:
                    horizon = horizon_2
                    trajectory = trajectory_2
                for agent in agents:
                    agent_init_euler = (0, 0, 0)
                    data_path = os.path.join(script_path, '..', f'data-motion-{motion_ind}', f'eef-{agent}')
                    if agent == 'rectangle':
                        agent_init_euler = (0, 0, 45)

                    for data_ind in data_inds:
                        ti.reset()
                        ti.init(arch=ti.opengl, default_fp=DTYPE_TI, default_ip=ti.i32, fast_math=False, random_seed=seed,
                                debug=True, check_out_of_bound=True)
                        print(f'=====> Computing: epoch {epoch}, motion {motion_ind}, agent {agent}, data {data_ind}')
                        env, mpm_env, init_state = make_env(data_path, str(data_ind), horizon,
                                                            particle_density, DTYPE_NP,
                                                            agent, agent_init_euler, loss_cfg.copy())
                        set_parameters(mpm_env, E.copy(), nu.copy(), yield_stress.copy())
                        loss_info = forward_backward(mpm_env, init_state, trajectory, render=False)
                        for i, v in loss.items():
                            loss[i] += loss_info[i]
                        grad = np.array([mpm_env.simulator.particle_param.grad[MATERIAL_ID].E,
                                         mpm_env.simulator.particle_param.grad[MATERIAL_ID].nu,
                                         mpm_env.simulator.system_param.grad[None].yield_stress], dtype=DTYPE_NP)
                        print(f'=====> Total loss: {mpm_env.loss.total_loss[None]}')
                        print(f'=====> Grad: {grad}')
                        grads += grad

            for i, v in loss.items():
                loss[i] = v / num_datapoints
            grads = grads / num_datapoints

            E = optim_E.step(E.copy(), grads[0])
            E = np.clip(E, E_range[0], E_range[1])
            nu = optim_nu.step(nu.copy(), grads[1])
            nu = np.clip(nu, nu_range[0], nu_range[1])
            yield_stress = optim_yield_stress.step(yield_stress.copy(), grads[2])
            yield_stress = np.clip(yield_stress, yield_stress_range[0], yield_stress_range[1])

            for i, v in loss.items():
                logger.add_scalar(tag=f'Loss/{i}', scalar_value=v, global_step=epoch)
            logger.add_scalar(tag='Param/E', scalar_value=E, global_step=epoch)
            logger.add_scalar(tag='Grad/E', scalar_value=grads[0], global_step=epoch)
            logger.add_scalar(tag='Param/nu', scalar_value=nu, global_step=epoch)
            logger.add_scalar(tag='Grad/nu', scalar_value=grads[1], global_step=epoch)
            logger.add_scalar(tag='Param/yield_stress', scalar_value=yield_stress, global_step=epoch)
            logger.add_scalar(tag='Grad/yield_stress', scalar_value=grads[2], global_step=epoch)
            logger.add_scalar(tag='Mem/GPU', scalar_value=get_gpu_memory()[-1], global_step=epoch)
            logger.add_scalar(tag='Mem/RAM', scalar_value=process.memory_percent(), global_step=epoch)
            logger.flush()
            print(f"========> Epoch {epoch}: time={time() - t1}\n"
                  f"========> E={E}, nu={nu}, yield_stress={yield_stress}")
            for i, v in loss.items():
                print(f"========> Loss: {i}: {v}")

        logger.close()
        print(f"Final parameters: E={E}, nu={nu}, yield_stress={yield_stress}")
        np.save(os.path.join(log_dir, 'final_params.npy'), np.array([E, nu, yield_stress], dtype=DTYPE_NP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adam', dest='adam', default=False, action='store_true')
    parser.add_argument('--pd_rs_loss', dest='pd_rs_loss', default=False, action='store_true')
    parser.add_argument('--pd_sr_loss', dest='pd_sr_loss', default=False, action='store_true')
    parser.add_argument('--prd_rs_loss', dest='prd_rs_loss', default=False, action='store_true')
    parser.add_argument('--prd_sr_loss', dest='prd_sr_loss', default=False, action='store_true')
    parser.add_argument('--hm_loss', dest='hm_loss', default=False, action='store_true')
    parser.add_argument('--hm_res', dest='hm_res', default=32, type=int)
    args = vars(parser.parse_args())
    main(args)
