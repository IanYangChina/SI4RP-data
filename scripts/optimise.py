import os
import numpy as np
from time import time
from doma.envs import SysIDEnv
from torch.utils.tensorboard import SummaryWriter


MATERIAL_ID = 2
DTYPE_NP = np.float32
# Optimisation scheme: Adam
# Optimisation objective: Chamfer distance
# Schedule:
#       compute forward-backward 5 times for each object configuration
#       update parameters once every forward simulation trajectory, lr = 0.001


def forward_backward(mpm_env, init_state, trajectory, render):
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
    print(f'Final chamfer loss: {mpm_env.loss.total_loss[None]}')

    t2 = time()

    # backward
    mpm_env.reset_grad()
    mpm_env.get_final_loss_grad()
    for i in range(mpm_env.horizon-1, -1, -1):
        action = trajectory[i]
        mpm_env.step_grad(action=action)
    # mpm_env.apply_agent_action_p_grad(np.array([0.25, 0.25, 2*centre_top_normalised[-1]+0.01, 0, 0, 45]))

    t3 = time()
    print(f'=======> forward: {t2 - t1:.2f}s backward: {t3 - t2:.2f}s')


def make_env(data_path, pcd_ind_start, pcd_ind_end, horizon):
    obj_start_mesh_file_path = os.path.join(data_path, 'mesh_' + pcd_ind_start + '_repaired_normalised.obj')
    if not os.path.exists(obj_start_mesh_file_path):
        return None, None
    obj_start_centre_real = np.load(os.path.join(data_path, 'mesh_' + pcd_ind_start + '_repaired_centre.npy'))
    obj_start_centre_top_normalised = np.load(os.path.join(data_path, 'mesh_' + pcd_ind_start + '_repaired_normalised_centre_top.npy'))
    obj_end_pcd_file_path = os.path.join(data_path, 'pcd_' + pcd_ind_end + '.ply')
    if not os.path.exists(obj_end_pcd_file_path):
        return None, None
    obj_end_mesh_file_path = os.path.join(data_path, 'mesh_' + pcd_ind_end + '_repaired_normalised.obj')
    obj_end_centre_top_normalised = np.load(os.path.join(data_path, 'mesh_' + pcd_ind_end + '_repaired_normalised_centre_top.npy'))

    # Building environment
    obj_start_initial_pos = np.array([0.25, 0.25, obj_start_centre_top_normalised[-1] + 0.01], dtype=DTYPE_NP)

    env = SysIDEnv(ptcl_density=1e7, horizon=horizon, material_id=MATERIAL_ID, voxelise_res=1080,
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


def set_parameters(mpm_env, E, nu, yield_stress):
    mpm_env.simulator.system_param_tmp[None].yield_stress = yield_stress
    mpm_env.simulator.particle_param_tmp[MATERIAL_ID].E = E
    mpm_env.simulator.particle_param_tmp[MATERIAL_ID].nu = nu
    for s in range(mpm_env.horizon):
        mpm_env.simulator.system_param[s].yield_stress = yield_stress
        mpm_env.simulator.particle_param[s, MATERIAL_ID].E = E
        mpm_env.simulator.particle_param[s, MATERIAL_ID].nu = nu


def optimise(mpm_env, global_step, n_iter, init_state, trajectory,
             E, nu, yield_stress, E_lr, nu_lr, yield_stress_lr, logger):
    for i in range(n_iter):
        set_parameters(mpm_env, E, nu, yield_stress)
        forward_backward(mpm_env, init_state, trajectory, render=False)
        logger.add_scalar(tag='Loss/chamfer', scalar_value=mpm_env.loss.total_loss[None],
                          global_step=global_step+i)
        logger.add_scalar(tag='Param/E', scalar_value=E,
                          global_step=global_step+i)
        logger.add_scalar(tag='Param/nu', scalar_value=nu,
                          global_step=global_step+i)
        logger.add_scalar(tag='Param/yield_stress', scalar_value=yield_stress,
                          global_step=global_step+i)
        E -= E_lr * mpm_env.simulator.particle_param_tmp.grad[MATERIAL_ID].E
        nu -= nu_lr * mpm_env.simulator.particle_param_tmp.grad[MATERIAL_ID].nu
        yield_stress -= yield_stress_lr * mpm_env.simulator.system_param_tmp.grad[None].yield_stress


def main():
    script_path = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(script_path, '..', 'data-motion-1', 'optimisation_logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = SummaryWriter(log_dir=log_dir)

    # Setting up horizon and trajectory
    horizon = int(0.03 / 0.002)
    v = 0.045 / 0.03  # 1.5 m/s
    trajectory = np.zeros(shape=(horizon, 6), dtype=DTYPE_NP)
    trajectory[:5, 2] = -v
    trajectory[5:, 2] = v

    # Initialising parameters
    e = np.array([100], dtype=DTYPE_NP)  # Young's modulus
    e_lr = 1e-3
    nu = np.array([0.2], dtype=DTYPE_NP)  # Poisson's ratio
    nu_lr = 1e-3
    yield_stress = np.array([1], dtype=DTYPE_NP)
    yield_stress_lr = 1e-3

    global_step = 0
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
                    optimise(mpm_env, global_step=global_step, n_iter=10, init_state=init_state, trajectory=trajectory,
                             E=e, nu=nu, yield_stress=yield_stress,
                             E_lr=e_lr, nu_lr=nu_lr, yield_stress_lr=yield_stress_lr, logger=logger)
                    global_step += 10
            trial_id += 1
            data_path = os.path.join(script_path, '..', 'data-motion-1', f'trial-{trial_id}')


if __name__ == '__main__':
    main()
