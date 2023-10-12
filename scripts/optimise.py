import os
import taichi as ti
import numpy as np
from time import time
from torch.utils.tensorboard import SummaryWriter

MATERIAL_ID = 2
# Optimisation scheme: Adam
# Optimisation objective: Chamfer distance
# Schedule:
#       compute forward-backward 5 times for each object configuration
#       update parameters once every forward simulation trajectory, lr = 0.001


def forward_backward(mpm_env, init_state, trajectory, render, backward=True):
    # Forward
    t1 = time()
    mpm_env.set_state(init_state['state'], grad_enabled=True)
    # mpm_env.apply_agent_action_p(np.array([0.25, 0.25, 2*centre_top_normalised[-1]+0.01, 0, 0, 45]))
    for i in range(mpm_env.horizon):
        action = trajectory[i]
        mpm_env.step(action)
        if render:
            mpm_env.render(mode='human')
    mpm_env.get_final_loss()
    print(
        f'Final chamfer loss: {mpm_env.loss.total_loss[None]}')
    t2 = time()

    if backward:
        # backward
        mpm_env.reset_grad()
        mpm_env.get_final_loss_grad()
        for i in range(mpm_env.horizon - 1, -1, -1):
            action = trajectory[i]
            mpm_env.step_grad(action=action)
        # mpm_env.apply_agent_action_p_grad(np.array([0.25, 0.25, 2*centre_top_normalised[-1]+0.01, 0, 0, 45]))

        t3 = time()
        print(f'=======> forward: {t2 - t1:.2f}s backward: {t3 - t2:.2f}s')


def set_parameters(mpm_env, E, nu, yield_stress):
    mpm_env.simulator.system_param_tmp[None].yield_stress = yield_stress.copy()
    mpm_env.simulator.particle_param_tmp[MATERIAL_ID].E = E.copy()
    mpm_env.simulator.particle_param_tmp[MATERIAL_ID].nu = nu.copy()
    for s in range(mpm_env.horizon):
        mpm_env.simulator.system_param[s].yield_stress = yield_stress.copy()
        mpm_env.simulator.particle_param[s, MATERIAL_ID].E = E.copy()
        mpm_env.simulator.particle_param[s, MATERIAL_ID].nu = nu.copy()


def optimise(mpm_env, global_step, n_iter, init_state, trajectory,
             E, nu, yield_stress, E_lr, nu_lr, yield_stress_lr, logger):
    E_ = E.copy()
    nu_ = nu.copy()
    yield_stress_ = yield_stress.copy()
    for i in range(n_iter):
        set_parameters(mpm_env, E_, nu_, yield_stress_)
        forward_backward(mpm_env, init_state, trajectory, render=False)
        logger.add_scalar(tag='Loss/chamfer', scalar_value=mpm_env.loss.total_loss[None],
                          global_step=global_step + i)
        logger.add_scalar(tag='Param/E', scalar_value=E_,
                          global_step=global_step + i)
        logger.add_scalar(tag='Param/nu', scalar_value=nu_,
                          global_step=global_step + i)
        logger.add_scalar(tag='Param/yield_stress', scalar_value=yield_stress_,
                          global_step=global_step + i)

        E_ = E_ - E_lr * mpm_env.simulator.particle_param_tmp.grad[MATERIAL_ID].E
        E_ = np.clip(E_, 1, 500)
        logger.add_scalar(tag='Grad/E', scalar_value=mpm_env.simulator.particle_param_tmp.grad[MATERIAL_ID].E,
                          global_step=global_step + i)
        nu_ = nu_ - nu_lr * mpm_env.simulator.particle_param_tmp.grad[MATERIAL_ID].nu
        nu_ = np.clip(nu_, 0.0001, 0.9999)
        logger.add_scalar(tag='Grad/nu', scalar_value=mpm_env.simulator.particle_param_tmp.grad[MATERIAL_ID].nu,
                          global_step=global_step + i)
        yield_stress_ = yield_stress_ - yield_stress_lr * mpm_env.simulator.system_param_tmp.grad[None].yield_stress
        yield_stress_ = np.clip(yield_stress_, 0.0001, 4)
        logger.add_scalar(tag='Grad/yield_stress',
                          scalar_value=mpm_env.simulator.system_param_tmp.grad[None].yield_stress,
                          global_step=global_step + i)
    return E_, nu_, yield_stress_


def main():
    script_path = os.path.dirname(os.path.realpath(__file__))
    DTYPE_NP = np.float32
    DTYPE_TI = ti.f32

    # Setting up horizon and trajectory
    horizon = int(0.03 / 0.002)
    v = 0.045 / 0.03  # 1.5 m/s
    trajectory = np.zeros(shape=(horizon, 6), dtype=DTYPE_NP)
    trajectory[:5, 2] = -v
    trajectory[5:, 2] = v

    # Learning rates
    e_lr = 0.1
    nu_lr = 0.0001
    yield_stress_lr = 0.0001

    seeds = [0, 1]
    for seed in seeds:
        # Setting up random seed
        np.random.seed(seed)

        def make_env(data_path, pcd_ind_start, pcd_ind_end, horizon):
            ti.init(arch=ti.vulkan, device_memory_GB=5, default_fp=DTYPE_TI, fast_math=False, random_seed=seed)
            from doma.envs import SysIDEnv

            obj_start_mesh_file_path = os.path.join(data_path, 'mesh_' + pcd_ind_start + '_repaired_normalised.obj')
            if not os.path.exists(obj_start_mesh_file_path):
                return None, None
            obj_start_centre_real = np.load(os.path.join(data_path, 'mesh_' + pcd_ind_start + '_repaired_centre.npy'))
            obj_start_centre_top_normalised = np.load(
                os.path.join(data_path, 'mesh_' + pcd_ind_start + '_repaired_normalised_centre_top.npy'))
            obj_end_pcd_file_path = os.path.join(data_path, 'pcd_' + pcd_ind_end + '.ply')
            if not os.path.exists(obj_end_pcd_file_path):
                return None, None
            obj_end_mesh_file_path = os.path.join(data_path, 'mesh_' + pcd_ind_end + '_repaired_normalised.obj')
            obj_end_centre_top_normalised = np.load(
                os.path.join(data_path, 'mesh_' + pcd_ind_end + '_repaired_normalised_centre_top.npy'))

            # Building environment
            obj_start_initial_pos = np.array([0.25, 0.25, obj_start_centre_top_normalised[-1] + 0.01], dtype=DTYPE_NP)

            env = SysIDEnv(ptcl_density=2e7, horizon=horizon, material_id=MATERIAL_ID, voxelise_res=1080,
                           mesh_file=obj_start_mesh_file_path, initial_pos=obj_start_initial_pos,
                           target_pcd_file=obj_end_pcd_file_path,
                           pcd_offset=(-obj_start_centre_real + obj_start_initial_pos),
                           target_mesh_file=obj_end_mesh_file_path,
                           mesh_offset=(0.25, 0.25, obj_end_centre_top_normalised[-1] + 0.01),
                           loss_weight=1.0, separate_param_grad=False)
            env.reset()
            mpm_env = env.mpm_env
            init_state = mpm_env.get_state()

            return mpm_env, init_state

        # Logger
        log_dir = os.path.join(script_path, '..', 'data-motion-1', 'optimisation-logs', f'seed-{seed}')
        os.makedirs(log_dir, exist_ok=True)
        logger = SummaryWriter(log_dir=log_dir)

        # Initialising parameters
        e = np.asarray(np.random.uniform(1, 500), dtype=DTYPE_NP)  # Young's modulus
        nu = np.asarray(np.random.uniform(0.0001, 0.9999), dtype=DTYPE_NP)  # Poisson's ratio
        yield_stress = np.asarray(np.random.uniform(0.0001, 4), dtype=DTYPE_NP)

        global_step = 0
        n_local_step = 40
        for i in range(3):
            trial_id = 1
            data_path = os.path.join(script_path, '..', 'data-motion-1', f'trial-{trial_id}')

            while True:
                if not os.path.exists(data_path):
                    print('path not exist')
                    break
                else:
                    for pcd_ind_start in range(4):
                        pcd_ind_end = pcd_ind_start + 1
                        mpm_env, init_state = make_env(data_path, str(pcd_ind_start), str(pcd_ind_end), horizon)
                        if mpm_env is None:
                            continue
                        else:
                            print(f'Optimising with trial-{trial_id} pcd-{pcd_ind_start}-{pcd_ind_end}')
                            e_, nu_, yield_stress_ = optimise(mpm_env, global_step=global_step, n_iter=n_local_step,
                                                              init_state=init_state, trajectory=trajectory,
                                                              E=e, nu=nu, yield_stress=yield_stress,
                                                              E_lr=e_lr, nu_lr=nu_lr, yield_stress_lr=yield_stress_lr,
                                                              logger=logger)
                            e = e_.copy()
                            nu = nu_.copy()
                            yield_stress = yield_stress_.copy()

                            global_step += n_local_step
                            del mpm_env, init_state

                    trial_id += 1
                    data_path = os.path.join(script_path, '..', 'data-motion-1', f'trial-{trial_id}')

        # Evaluation
        print(f'Perform evaluation with final parameters: E={e}, nu={nu}, yield_stress={yield_stress}')
        trial_id = 1
        data_path = os.path.join(script_path, '..', 'data-motion-1', f'trial-{trial_id}')
        while True:
            if not os.path.exists(data_path):
                print('path not exist')
                break
            else:
                for pcd_ind_start in range(4):
                    pcd_ind_end = pcd_ind_start + 1
                    mpm_env, init_state = make_env(data_path, str(pcd_ind_start), str(pcd_ind_end), horizon)
                    if mpm_env is None:
                        continue
                    else:
                        print(f'Optimising with trial-{trial_id} pcd-{pcd_ind_start}-{pcd_ind_end}')
                        set_parameters(mpm_env, e, nu, yield_stress)
                        forward_backward(mpm_env, init_state, trajectory, render=True, backward=False)


if __name__ == '__main__':
    main()
