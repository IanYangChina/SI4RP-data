import os
import argparse
import taichi as ti
import numpy as np
from time import time
from torch.utils.tensorboard import SummaryWriter
from doma.optimiser.adam import Adam, GD
from doma.envs.sys_id_env import make_env, set_parameters
from doma.engine.utils.misc import get_gpu_memory
import psutil
import json

MATERIAL_ID = 2
process = psutil.Process(os.getpid())


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

def main(arguments):
    script_path = os.path.dirname(os.path.realpath(__file__))
    DTYPE_NP = np.float32
    DTYPE_TI = ti.f32
    particle_density = 3e7
    assert arguments['hm_res'] in [32, 64], 'height map resolution must be 32 or 64'
    loss_cfg = {
        'exponential_distance': arguments['exp_dist'],
        'averaging_loss': arguments['averaging_loss'],
        'point_distance_rs_loss': arguments['pd_rs_loss'],
        'point_distance_sr_loss': arguments['pd_sr_loss'],
        'down_sample_voxel_size': 0.006,
        'particle_distance_rs_loss': arguments['prd_rs_loss'],
        'particle_distance_sr_loss': arguments['prd_sr_loss'],
        'voxelise_res': 1080,
        'ptcl_density': particle_density,
        'load_height_map': True,
        'height_map_loss': arguments['hm_loss'],
        'height_map_res': arguments['hm_res'],
        'height_map_size': 0.11,
        'emd_point_distance_loss': arguments['emd_point_distance_loss'],
    }

    # Setting up horizon and trajectory.

    # Parameter ranges
    E_range = (10000, 100000)
    nu_range = (0.001, 0.49)
    yield_stress_range = (50, 3000)

    n_epoch = 100
    n_aborted_data = 0
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
        json.dump(loss_cfg, f_ac, indent=2)

    training_config = {
        'lr_E': 1e3,
        'lr_nu': 1e-2,
        'lr_yield_stress': 1e2,
        'batch_size': arguments['batchsize'],
        'n_epoch': n_epoch,
        'seeds': seeds,
    }
    with open(os.path.join(log_p_dir, 'training_config.json'), 'w') as f_ac:
        json.dump(training_config, f_ac, indent=2)

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
        # Optimiser: Adam
        optim_E = Adam(parameters_shape=E.shape,
                       cfg={'lr': training_config['lr_E'], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
        optim_nu = Adam(parameters_shape=nu.shape,
                        cfg={'lr': training_config['lr_nu'], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
        optim_yield_stress = Adam(parameters_shape=yield_stress.shape,
                                  cfg={'lr': training_config['lr_yield_stress'], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})

        agents = ['rectangle', 'round', 'cylinder']
        mini_batch_size = arguments['batchsize']

        for epoch in range(n_epoch):
            t1 = time()
            loss = {
                'avg_point_distance_sr': [],
                'avg_point_distance_rs': [],
                'chamfer_loss_pcd': [],
                'avg_particle_distance_sr': [],
                'avg_particle_distance_rs': [],
                'chamfer_loss_particle': [],
                'height_map_loss_pcd': [],
                'emd_loss': [],
                'total_loss': []
            }
            grads = []
            motion_ids = np.random.randint(1, 3, size=mini_batch_size, dtype=np.int32).tolist()
            agent_ids = np.random.randint(3, size=mini_batch_size, dtype=np.int32).tolist()
            data_ids = np.random.randint(9, size=mini_batch_size, dtype=np.int32).tolist()

            for i in range(mini_batch_size):
                motion_ind = str(motion_ids[i])
                trajectory = np.load(os.path.join(script_path, '..', f'data-motion-{motion_ind}', 'tr_eef_v.npy'))
                dt_global = np.load(os.path.join(script_path, '..', f'data-motion-{motion_ind}', 'tr_dt.npy'))
                horizon = trajectory.shape[0]
                n_substeps = 50

                agent = agents[agent_ids[i]]
                agent_init_euler = (0, 0, 0)
                training_data_path = os.path.join(script_path, '..', f'data-motion-{motion_ind}', f'eef-{agent}')
                if agent == 'rectangle':
                    agent_init_euler = (0, 0, 45)

                data_ind = data_ids[i]

                ti.reset()
                ti.init(arch=ti.opengl, default_fp=DTYPE_TI, default_ip=ti.i32, fast_math=False, random_seed=seed,
                        debug=False, check_out_of_bound=False)
                data_cfg = {
                    'data_path': training_data_path,
                    'data_ind': str(data_ind),
                }
                env_cfg = {
                    'p_density': particle_density,
                    'horizon': horizon,
                    'dt_global': dt_global,
                    'n_substeps': n_substeps,
                    'material_id': 2,
                    'agent_name': agent,
                    'agent_init_euler': agent_init_euler,
                }
                print(f'=====> Computing: epoch {epoch}, motion {motion_ind}, agent {agent}, data {data_ind}')
                env, mpm_env, init_state = make_env(data_cfg, env_cfg, loss_cfg)
                set_parameters(mpm_env, env_cfg['material_id'],
                               E.copy(), nu.copy(), yield_stress.copy(), rho=1000)
                loss_info = forward_backward(mpm_env, init_state, trajectory, render=False)
                abort = False
                if loss_info['total_loss'] < 1e-20:
                    abort = True
                if (loss_info['total_loss'] > 100) and arguments['averaging_loss']:
                    abort = True
                if (loss_info['total_loss'] > 40000) and (not arguments['averaging_loss']):
                    abort = True
                if (np.isinf(loss_info['height_map_loss_pcd'])) or (np.isnan(loss_info['emd_loss'])):
                    abort = True

                grad = np.array([mpm_env.simulator.particle_param.grad[MATERIAL_ID].E,
                                 mpm_env.simulator.particle_param.grad[MATERIAL_ID].nu,
                                 mpm_env.simulator.system_param.grad[None].yield_stress], dtype=DTYPE_NP)
                print(f'=====> Total loss: {mpm_env.loss.total_loss[None]}')
                print(f'=====> Grad: {grad}')

                if not abort:
                    for i, v in loss.items():
                        loss[i].append(loss_info[i])
                    grads.append(grad.copy())
                else:
                    n_aborted_data += 1
                    print(f'=====> Aborting datapoint: epoch {epoch}, motion {motion_ind}, agent {agent}, data {data_ind}')
                mpm_env.simulator.clear_ckpt()

            for i, v in loss.items():
                loss[i] = np.mean(v)
            avg_grad = np.mean(grads, axis=0)

            E = optim_E.step(E.copy(), avg_grad[0])
            E = np.clip(E, E_range[0], E_range[1])
            nu = optim_nu.step(nu.copy(), avg_grad[1])
            nu = np.clip(nu, nu_range[0], nu_range[1])
            yield_stress = optim_yield_stress.step(yield_stress.copy(), avg_grad[2])
            yield_stress = np.clip(yield_stress, yield_stress_range[0], yield_stress_range[1])

            for i, v in loss.items():
                logger.add_scalar(tag=f'Loss/{i}', scalar_value=v, global_step=epoch)
            logger.add_scalar(tag='Param/E', scalar_value=E, global_step=epoch)
            logger.add_scalar(tag='Grad/E', scalar_value=avg_grad[0], global_step=epoch)
            logger.add_scalar(tag='Param/nu', scalar_value=nu, global_step=epoch)
            logger.add_scalar(tag='Grad/nu', scalar_value=avg_grad[1], global_step=epoch)
            logger.add_scalar(tag='Param/yield_stress', scalar_value=yield_stress, global_step=epoch)
            logger.add_scalar(tag='Grad/yield_stress', scalar_value=avg_grad[2], global_step=epoch)
            logger.add_scalar(tag='Mem/GPU', scalar_value=get_gpu_memory()[0], global_step=epoch)
            logger.add_scalar(tag='Mem/RAM', scalar_value=process.memory_percent(), global_step=epoch)
            logger.add_scalar(tag='Aborted', scalar_value=n_aborted_data, global_step=epoch)

            print(f"========> Epoch {epoch}: time={time() - t1}\n"
                  f"========> E={E}, nu={nu}, yield_stress={yield_stress}")
            for i, v in loss.items():
                print(f"========> Loss: {i}: {v}")
            print(f"========> Avg. grads: {grads}")

        logger.close()
        print(f"Final parameters: E={E}, nu={nu}, yield_stress={yield_stress}")
        np.save(os.path.join(log_dir, 'final_params.npy'), np.array([E, nu, yield_stress], dtype=DTYPE_NP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dist', dest='exp_dist', default=False, action='store_true')
    parser.add_argument('--avg_loss', dest='averaging_loss', default=False, action='store_true')
    parser.add_argument('--pd_rs_loss', dest='pd_rs_loss', default=False, action='store_true')
    parser.add_argument('--pd_sr_loss', dest='pd_sr_loss', default=False, action='store_true')
    parser.add_argument('--prd_rs_loss', dest='prd_rs_loss', default=False, action='store_true')
    parser.add_argument('--prd_sr_loss', dest='prd_sr_loss', default=False, action='store_true')
    parser.add_argument('--emd_loss', dest='emd_point_distance_loss', default=False, action='store_true')
    parser.add_argument('--hm_loss', dest='hm_loss', default=False, action='store_true')
    parser.add_argument('--hm_res', dest='hm_res', default=32, type=int)
    parser.add_argument('--bs', dest='batchsize', default=20, type=int)
    args = vars(parser.parse_args())
    main(args)
