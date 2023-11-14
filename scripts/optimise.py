import os
import argparse
import taichi as ti
import numpy as np
from time import time
from torch.utils.tensorboard import SummaryWriter
from doma.optimiser.adam import Adam, SGD
from doma.engine.utils.misc import get_gpu_memory

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
    mpm_env.get_final_loss()

    t2 = time()

    if backward:
        # backward
        mpm_env.reset_grad()
        mpm_env.get_final_loss_grad()
        for i in range(mpm_env.horizon - 1, -1, -1):
            action = trajectory[i]
            mpm_env.step_grad(action=action)

        t3 = time()
        print(f'=====> forward: {t2 - t1:.2f}s backward: {t3 - t2:.2f}s')


def set_parameters(mpm_env, E, nu, yield_stress):
    mpm_env.simulator.system_param[None].yield_stress = yield_stress
    mpm_env.simulator.particle_param[MATERIAL_ID].rho = 1000
    mpm_env.simulator.particle_param[MATERIAL_ID].E = E
    mpm_env.simulator.particle_param[MATERIAL_ID].nu = nu


def main(arguments):
    script_path = os.path.dirname(os.path.realpath(__file__))
    DTYPE_NP = np.float32
    DTYPE_TI = ti.f32
    ptcl_density = 4e7

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
    for seed in seeds:
        # Setting up random seed
        np.random.seed(seed)

        def make_env(data_path, data_ind, horizon, agent_name, agent_init_euler):
            ti.init(arch=ti.vulkan, device_memory_GB=10, default_fp=DTYPE_TI, fast_math=False, random_seed=seed)
            from doma.envs import SysIDEnv
            obj_start_mesh_file_path = os.path.join(data_path, 'mesh_' + data_ind+str(0) + '_repaired_normalised.obj')
            if not os.path.exists(obj_start_mesh_file_path):
                return None, None
            obj_start_centre_real = np.load(os.path.join(data_path, 'mesh_' + data_ind+str(0) + '_repaired_centre.npy')).astype(DTYPE_NP)
            obj_start_centre_top_normalised = np.load(
                os.path.join(data_path, 'mesh_' + data_ind+str(0) + '_repaired_normalised_centre_top.npy')).astype(DTYPE_NP)

            obj_end_pcd_file_path = os.path.join(data_path, 'pcd_' + data_ind+str(1) + '.ply')
            obj_end_mesh_file_path = os.path.join(data_path, 'mesh_' + data_ind+str(1) + '_repaired_normalised.obj')
            obj_end_centre_top_normalised = np.load(
                os.path.join(data_path, 'mesh_' + data_ind+str(1) + '_repaired_normalised_centre_top.npy')).astype(DTYPE_NP)

            # Building environment
            obj_start_initial_pos = np.array([0.25, 0.25, obj_start_centre_top_normalised[-1] + 0.01], dtype=DTYPE_NP)
            agent_init_pos = (0.25, 0.25, 2*obj_start_centre_top_normalised[-1] + 0.01)

            env = SysIDEnv(ptcl_density=ptcl_density, horizon=horizon, material_id=MATERIAL_ID, voxelise_res=1080,
                           mesh_file=obj_start_mesh_file_path, initial_pos=obj_start_initial_pos,
                           target_pcd_file=obj_end_pcd_file_path, down_sample_voxel_size=0.0035,
                           pcd_offset=(-obj_start_centre_real + obj_start_initial_pos),
                           target_mesh_file=obj_end_mesh_file_path,
                           mesh_offset=(0.25, 0.25, obj_end_centre_top_normalised[-1] + 0.01),
                           loss_weight=1.0, separate_param_grad=False,
                           agent_cfg_file=agent_name+'_eef.yaml', agent_init_pos=agent_init_pos, agent_init_euler=agent_init_euler)
            env.reset()
            mpm_env = env.mpm_env
            init_state = mpm_env.get_state()

            return env, mpm_env, init_state

        # Logger
        log_dir = os.path.join(script_path, '..', 'optimisation-logs', f'seed-{seed}')
        os.makedirs(log_dir, exist_ok=True)
        logger = SummaryWriter(log_dir=log_dir)

        # Initialising parameters
        E = np.asarray(np.random.uniform(E_range[0], E_range[1]), dtype=DTYPE_NP).reshape((1,))  # Young's modulus
        nu = np.asarray(np.random.uniform(nu_range[0], nu_range[1]), dtype=DTYPE_NP).reshape((1,))  # Poisson's ratio
        yield_stress = np.asarray(np.random.uniform(yield_stress_range[0], yield_stress_range[1]), dtype=DTYPE_NP).reshape((1,))  # Yield stress

        print(f"=====> Seed: {seed}, initial parameters: E={E}, nu={nu}, yield_stress={yield_stress}")
        if arguments['adam']:
            # Optimiser: Adam
            optim_E = Adam(parameters_shape=E.shape,
                          cfg={'lr': 1e9, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
            optim_nu = Adam(parameters_shape=nu.shape,
                           cfg={'lr': 0.1, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
            optim_yield_stress = Adam(parameters_shape=yield_stress.shape,
                                     cfg={'lr': 1e6, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
        else:
            # Optimiser: SGD
            optim_E = SGD(parameters_shape=E.shape,
                         cfg={'lr': 1e9})
            optim_nu = SGD(parameters_shape=nu.shape,
                          cfg={'lr': 0.1})
            optim_yield_stress = SGD(parameters_shape=yield_stress.shape,
                                    cfg={'lr': 1e6})

        motion_inds = ['1', '2']
        agents = ['rectangle', 'round', 'cylinder']
        data_inds = np.arange(9).astype(int)
        num_datapoints = len(data_inds) * len(agents) * len(motion_inds)

        for epoch in range(n_epoch):
            t1 = time()
            loss = 0.0
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
                        print(f'=====> Computing: epoch {epoch}, motion {motion_ind}, agent {agent}, data {data_ind}')
#                        print(f'GPU memory before creating an env: {get_gpu_memory()}')
                        env, mpm_env, init_state = make_env(data_path, str(data_ind), horizon, agent, agent_init_euler)
#                        print(f'GPU memory after creating an env: {get_gpu_memory()}')
                        set_parameters(mpm_env, E.copy(), nu.copy(), yield_stress.copy())
                        forward_backward(mpm_env, init_state, trajectory, render=False)
                        loss += mpm_env.loss.total_loss[None]
                        grad = np.array([mpm_env.simulator.particle_param.grad[MATERIAL_ID].E,
                                           mpm_env.simulator.particle_param.grad[MATERIAL_ID].nu,
                                           mpm_env.simulator.system_param.grad[None].yield_stress], dtype=DTYPE_NP)
                        print(f'=====> Loss: {mpm_env.loss.total_loss[None]}')
                        print(f'=====> Grad: {grad}')
                        grads += grad
#                        print(f'GPU memory after forward-backward computation: {get_gpu_memory()}')
#                        del env, mpm_env, init_state
#                        print(f'GPU memory after deleting env: {get_gpu_memory()}')

            loss = loss / num_datapoints
            grads = grads / num_datapoints

            E = optim_E.step(E.copy(), grads[0])
            E = np.clip(E, E_range[0], E_range[1])
            nu = optim_nu.step(nu.copy(), grads[1])
            nu = np.clip(nu, nu_range[0], nu_range[1])
            yield_stress = optim_yield_stress.step(yield_stress.copy(), grads[2])
            yield_stress = np.clip(yield_stress, yield_stress_range[0], yield_stress_range[1])

            logger.add_scalar(tag='Loss/chamfer', scalar_value=loss, global_step=epoch)
            logger.add_scalar(tag='Param/E', scalar_value=E, global_step=epoch)
            logger.add_scalar(tag='Grad/E', scalar_value=grads[0], global_step=epoch)
            logger.add_scalar(tag='Param/nu', scalar_value=nu, global_step=epoch)
            logger.add_scalar(tag='Grad/nu', scalar_value=grads[1], global_step=epoch)
            logger.add_scalar(tag='Param/yield_stress', scalar_value=yield_stress, global_step=epoch)
            logger.add_scalar(tag='Grad/yield_stress', scalar_value=grads[2], global_step=epoch)
            print(f"========> Epoch {epoch}: time={time() - t1}\n"
                  f"========> loss={loss}, E={E}, nu={nu}, yield_stress={yield_stress}")

        print(f"Final parameters: E={E}, nu={nu}, yield_stress={yield_stress}")
        np.save(os.path.join(log_dir, 'final_params.npy'), np.array([E, nu, yield_stress], dtype=DTYPE_NP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adam', dest='adam', default=False, action='store_true')
    args = vars(parser.parse_args())
    main(args)
