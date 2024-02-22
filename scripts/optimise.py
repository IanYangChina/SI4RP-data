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
import logging

MATERIAL_ID = 2
process = psutil.Process(os.getpid())


def forward_backward(mpm_env, init_state, trajectory, backward=True):
    # Forward
    mpm_env.set_state(init_state['state'], grad_enabled=True)
    for i in range(mpm_env.horizon):
        action = trajectory[i]
        mpm_env.step(action)

    loss_info = mpm_env.get_final_loss()

    if backward and (not loss_info['particle_has_naninf']):
        # backward
        mpm_env.reset_grad()
        mpm_env.get_final_loss_grad()
        for i in range(mpm_env.horizon - 1, -1, -1):
            action = trajectory[i]
            mpm_env.step_grad(action=action)

            # This is a trick that prevents faulty gradient computation
            # It works for unknown reasons
            _ = mpm_env.simulator.particle_param.grad[2].E

    return loss_info


def main(arguments):
    if arguments['backend'] == 'opengl':
        backend = ti.opengl
    elif arguments['backend'] == 'cuda':
        backend = ti.cuda
    elif arguments['backend'] == 'vulkan':
        backend = ti.vulkan
    else:
        backend = ti.cpu

    script_path = os.path.dirname(os.path.realpath(__file__))
    DTYPE_NP = np.float32
    DTYPE_TI = ti.f32
    particle_density = 4e7
    assert arguments['hm_res'] in [32, 64], 'height map resolution must be 32 or 64'
    loss_cfg = {
        'exponential_distance': arguments['exp_dist'],
        'averaging_loss': False,
        'point_distance_rs_loss': arguments['pd_rs_loss'],
        'point_distance_sr_loss': arguments['pd_sr_loss'],
        'down_sample_voxel_size': 0.005,
        'particle_distance_rs_loss': arguments['prd_rs_loss'],
        'particle_distance_sr_loss': arguments['prd_sr_loss'],
        'voxelise_res': 1080,
        'ptcl_density': particle_density,
        'load_height_map': True,
        'height_map_loss': arguments['hm_loss'],
        'height_map_res': arguments['hm_res'],
        'height_map_size': 0.11,
        'emd_point_distance_loss': arguments['emd_p_loss'],
        'emd_particle_distance_loss': arguments['emd_pr_loss'],
    }

    # Parameter ranges
    E_range = (1e4, 3e5)
    nu_range = (0.01, 0.48)
    yield_stress_range = (1e3, 2e4)
    rho_range = (1000, 2000)
    mf_range = (0.01, 2.0)
    gf_range = (0.01, 2.0)

    param_set = arguments['param_set']
    assert param_set in [0, 1], 'param_set must be 0 or 1'

    n_epoch = 100
    n_aborted_data = 0
    if arguments['seed'] != -1:
        seeds = [arguments['seed']]
    else:
        seeds = [0, 1, 2]
    if arguments['n_run'] != -1:
        # Append experiment into existing folder
        n = arguments['n_run']
        if arguments['fewshot']:
            log_p_dir = os.path.join(script_path, '..', f'optimisation-fewshot-param{param_set}-run{n}-logs')
        elif arguments['oneshot']:
            log_p_dir = os.path.join(script_path, '..', f'optimisation-oneshot-param{param_set}-run{n}-logs')
        else:
            log_p_dir = os.path.join(script_path, '..', f'optimisation-param{param_set}-run{n}-logs')
    else:
        # New experiments
        n = 0
        while True:
            if arguments['fewshot']:
                log_p_dir = os.path.join(script_path, '..', f'optimisation-fewshot-param{param_set}-run{n}-logs')
            elif arguments['oneshot']:
                log_p_dir = os.path.join(script_path, '..', f'optimisation-oneshot-param{param_set}-run{n}-logs')
            else:
                log_p_dir = os.path.join(script_path, '..', f'optimisation-param{param_set}-run{n}-logs')
            if os.path.exists(log_p_dir):
                n += 1
            else:
                break
    os.makedirs(log_p_dir, exist_ok=True)

    loss_cfg_file = os.path.join(log_p_dir, 'loss_config.json')
    if os.path.isfile(loss_cfg_file):
        with open(loss_cfg_file, 'r') as f_ac:
            loss_cfg = json.load(f_ac)
    else:
        with open(loss_cfg_file, 'w') as f_ac:
            json.dump(loss_cfg, f_ac, indent=2)

    log_file_name = os.path.join(log_p_dir, 'optimisation.log')
    if os.path.isfile(log_file_name):
        filemode = "a"
    else:
        filemode = "w"
    logging.basicConfig(level=logging.NOTSET, filemode=filemode,
                        filename=log_file_name,
                        format="%(asctime)s %(levelname)s %(message)s")

    training_config = {
        'param_set': param_set,
        'lr_E': 4e3,
        'lr_nu': 1e-2,
        'lr_yield_stress': 5e2,
        'lr_rho': 10,
        'lr_manipulator_friction': 1e-2,
        'lr_ground_friction': 1e-2,
        'batch_size': arguments['batchsize'],
        'n_epoch': n_epoch,
        'seeds': seeds,
    }
    if arguments['param_set'] == 1:
        training_config['lr_nu'] = 1e-3
        training_config['lr_yield_stress'] = 1e2
        training_config['lr_manipulator_friction'] = 0.1
        training_config['lr_ground_friction'] = 0.05
    training_config_file = os.path.join(log_p_dir, 'training_config.json')
    if os.path.isfile(training_config_file):
        with open(training_config_file, 'r') as f_ac:
            training_config = json.load(f_ac)
    else:
        with open(training_config_file, 'w') as f_ac:
            json.dump(training_config, f_ac, indent=2)

    print(f"=====> Optimising param set {param_set} for {n_epoch} epochs for {len(seeds)} random seeds.")
    print(f"=====> Loss config: {loss_cfg}")
    print(f"=====> Training config: {training_config}")
    logging.info(f"=====> Optimising param set {param_set} for {n_epoch} epochs for {len(seeds)} random seeds.")
    logging.info(f"=====> Loss config: {loss_cfg}")
    logging.info(f"=====> Training config: {training_config}")

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
        rho = np.asarray(np.random.uniform(rho_range[0], rho_range[1]),
                                  dtype=DTYPE_NP).reshape((1,))  # Density

        manipulator_friction = np.asarray([0.3], dtype=DTYPE_NP).reshape((1,))  # Manipulator friction
        ground_friction = np.asarray([2.0], dtype=DTYPE_NP).reshape((1,))  # Ground friction

        print(f"=====> Seed: {seed}, initial parameters: E={E}, nu={nu}, yield_stress={yield_stress}, rho={rho}")
        logging.info(f"=====> Seed: {seed}, initial parameters: E={E}, nu={nu}, yield_stress={yield_stress}, rho={rho}")
        # Optimiser: Adam
        optim_E = Adam(parameters_shape=E.shape,
                       cfg={'lr': training_config['lr_E'], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
        optim_nu = Adam(parameters_shape=nu.shape,
                        cfg={'lr': training_config['lr_nu'], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
        optim_yield_stress = Adam(parameters_shape=yield_stress.shape,
                                  cfg={'lr': training_config['lr_yield_stress'], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
        optim_rho = Adam(parameters_shape=rho.shape,
                                    cfg={'lr': training_config['lr_rho'], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
        if arguments['param_set'] == 1:
            # Initialising parameters
            # E = np.asarray([40000], dtype=DTYPE_NP).reshape((1,))  # Young's modulus
            # nu = np.asarray([0.4], dtype=DTYPE_NP).reshape((1,))
            # yield_stress = np.asarray([1000], dtype=DTYPE_NP).reshape((1,))
            # rho = np.asarray([1000], dtype=DTYPE_NP).reshape((1,))

            manipulator_friction = np.asarray(np.random.uniform(mf_range[0], mf_range[1]), dtype=DTYPE_NP).reshape((1,))  # Manipulator friction
            ground_friction = np.asarray(np.random.uniform(gf_range[0], gf_range[1]), dtype=DTYPE_NP).reshape((1,))

            print(f"=====> Seed: {seed}, initial parameters: manipulation friction={manipulator_friction}, ground friction={ground_friction}")
            logging.info(f"=====> Seed: {seed}, initial parameters: manipulation friction={manipulator_friction}, ground friction={ground_friction}")
            # Optimiser: Adam
            optim_mf = Adam(parameters_shape=manipulator_friction.shape,
                            cfg={'lr': training_config['lr_manipulator_friction'], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
            optim_gf = Adam(parameters_shape=ground_friction.shape,
                            cfg={'lr': training_config['lr_ground_friction'], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})

        agents = ['rectangle', 'round', 'cylinder']
        mini_batch_size = arguments['batchsize']

        for epoch in range(n_epoch):
            t1 = time()
            """===========Training==========="""
            loss = {
                'point_distance_sr': [],
                'point_distance_rs': [],
                'chamfer_loss_pcd': [],
                'particle_distance_sr': [],
                'particle_distance_rs': [],
                'chamfer_loss_particle': [],
                'height_map_loss_pcd': [],
                'emd_point_distance_loss': [],
                'emd_particle_distance_loss': [],
                'total_loss': []
            }
            grads = []
            if arguments['fewshot']:
                # data_id_dict = {
                #     '1': {'rectangle': [3, 5], 'round': [0, 1], 'cylinder': [1, 2]},
                #     '2': {'rectangle': [1, 3], 'round': [0, 2], 'cylinder': [0, 2]},
                #     '3': {'rectangle': [1, 2], 'round': [0, 1], 'cylinder': [0, 4]},
                #     '4': {'rectangle': [1, 3], 'round': [1, 4], 'cylinder': [0, 4]},
                # }
                mini_batch_size = 12
                if arguments['param_set'] == 0:
                    motion_ids = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
                    agent_ids = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
                    data_ids = [3, 5, 0, 1, 1, 2, 1, 3, 0, 2, 0, 2]
                else:
                    motion_ids = [3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]
                    agent_ids = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
                    data_ids = [1, 2, 0, 1, 0, 4, 1, 3, 1, 4, 0, 4]
            elif arguments['oneshot']:
                mini_batch_size = 6
                if arguments['param_set'] == 0:
                    motion_ids = [1, 1, 1, 2, 2, 2]
                    agent_ids = [0, 1, 2, 0, 1, 2]
                    data_ids = [3, 0, 1, 1, 0, 0]
                else:
                    motion_ids = [3, 3, 3, 4, 4, 4]
                    agent_ids = [0, 1, 2, 0, 1, 2]
                    data_ids = [1, 0, 0, 1, 1, 0]
            else:
                if arguments['param_set'] == 0:
                    motion_ids = np.random.randint(1, 3, size=mini_batch_size, dtype=np.int32).tolist()
                    data_ids = np.random.randint(9, size=mini_batch_size, dtype=np.int32).tolist()
                else:
                    motion_ids = np.random.randint(3, 5, size=mini_batch_size, dtype=np.int32).tolist()
                    data_ids = np.random.randint(5, size=mini_batch_size, dtype=np.int32).tolist()
                agent_ids = np.random.randint(3, size=mini_batch_size, dtype=np.int32).tolist()

            for i in range(mini_batch_size):
                motion_ind = str(motion_ids[i])
                dt_global = 0.01
                trajectory = np.load(os.path.join(script_path, '..', 'trajectories', f'tr_{motion_ind}_v_dt_{dt_global:0.2f}.npy'))
                horizon = trajectory.shape[0]
                n_substeps = 50

                agent = agents[agent_ids[i]]
                agent_init_euler = (0, 0, 0)
                if agent == 'rectangle':
                    agent_init_euler = (0, 0, 45)
                training_data_path = os.path.join(script_path, '..', f'data-motion-{motion_ind}', f'eef-{agent}')

                data_ind = data_ids[i]

                ti.reset()
                ti.init(arch=backend, default_fp=DTYPE_TI, default_ip=ti.i32, fast_math=True, random_seed=seed,
                        debug=False, check_out_of_bound=False, device_memory_GB=3)
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
                logging.info(f'=====> Computing: epoch {epoch}, motion {motion_ind}, agent {agent}, data {data_ind}')
                env, mpm_env, init_state = make_env(data_cfg, env_cfg, loss_cfg, logger=logging)
                set_parameters(mpm_env, env_cfg['material_id'],
                               E=E.copy(), nu=nu.copy(), yield_stress=yield_stress.copy(), rho=rho.copy(),
                               ground_friction=ground_friction.copy(),
                               manipulator_friction=manipulator_friction.copy())
                loss_info = forward_backward(mpm_env, init_state, trajectory)

                grad = np.array([mpm_env.simulator.particle_param.grad[MATERIAL_ID].E,
                                 mpm_env.simulator.particle_param.grad[MATERIAL_ID].nu,
                                 mpm_env.simulator.system_param.grad[None].yield_stress,
                                 mpm_env.simulator.particle_param.grad[MATERIAL_ID].rho,
                                 mpm_env.simulator.system_param.grad[None].manipulator_friction,
                                 mpm_env.simulator.system_param.grad[None].ground_friction], dtype=DTYPE_NP)
                # Check if the loss is strange
                abort = False
                particle_has_naninf = loss_info['particle_has_naninf']
                if particle_has_naninf:
                    abort = True

                if not abort:
                    for i, v in loss_info.items():
                        if i == 'final_height_map' or i == 'particle_has_naninf':
                            pass
                        else:
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
                    print(f'===> [Warning] Aborting datapoint: epoch {epoch}, motion {motion_ind}, agent {agent}, data {data_ind}')
                    print(f'===> [Warning] Particle has nan or inf: {particle_has_naninf}')
                    print(f'===> [Warning] Strange loss or gradient.')
                    print(f'===> [Warning] E: {E}, nu: {nu}, yield stress: {yield_stress}')
                    print(f'===> [Warning] Rho: {rho}, ground friction: {ground_friction}, manipulator friction: {manipulator_friction}')
                    logging.error(f'===> [Warning] Aborting datapoint: epoch {epoch}, motion {motion_ind}, agent {agent}, data {data_ind}')
                    logging.error(f'===> [Warning] Particle has nan or inf: {particle_has_naninf}')
                    logging.error(f'===> [Warning] Strange loss or gradient.')
                    logging.error(f'===> [Warning] E: {E}, nu: {nu}, yield stress: {yield_stress}')
                    logging.error(f'===> [Warning] Rho: {rho}, ground friction: {ground_friction}, manipulator friction: {manipulator_friction}')
                    n_aborted_data += 1
                else:
                    for i, v in loss.items():
                        loss[i].append(loss_info[i])
                    grads.append(grad.copy())

                print(f'=====> Total loss: {mpm_env.loss.total_loss[None]}')
                print(f'=====> Grad: {grad}')
                logging.info(f'=====> Total loss: {mpm_env.loss.total_loss[None]}')
                logging.info(f'=====> Grad: {grad}')

                mpm_env.simulator.clear_ckpt()

            """===========Statistics and updates==========="""
            for i, v in loss.items():
                loss[i] = np.mean(v)
            avg_grad = np.mean(grads, axis=0)

            for i, v in loss.items():
                print(f"========> Avg. Loss: {i}: {v}")
                logging.info(f"========> Avg. Loss: {i}: {v}")
                logger.add_scalar(tag=f'Loss/{i}', scalar_value=v, global_step=epoch)
            print(f"========> Avg. grads: {avg_grad}")
            print(f"========> Num. aborted data so far: {n_aborted_data}")
            logging.info(f"========> Avg. grads: {avg_grad}")
            logging.info(f"========> Num. aborted data so far: {n_aborted_data}")

            # Updates
            E = optim_E.step(E.copy(), avg_grad[0])
            E = np.clip(E, E_range[0], E_range[1])
            nu = optim_nu.step(nu.copy(), avg_grad[1])
            nu = np.clip(nu, nu_range[0], nu_range[1])
            yield_stress = optim_yield_stress.step(yield_stress.copy(), avg_grad[2])
            yield_stress = np.clip(yield_stress, yield_stress_range[0], yield_stress_range[1])
            rho = optim_rho.step(rho.copy(), avg_grad[3])
            rho = np.clip(rho, rho_range[0], rho_range[1])
            if arguments['param_set'] == 1:
                manipulator_friction = optim_mf.step(manipulator_friction.copy(), avg_grad[4])
                manipulator_friction = np.clip(manipulator_friction, mf_range[0], mf_range[1])
                ground_friction = optim_gf.step(ground_friction.copy(), avg_grad[5])
                ground_friction = np.clip(ground_friction, gf_range[0], gf_range[1])

            logger.add_scalar(tag='Param/E', scalar_value=E, global_step=epoch)
            logger.add_scalar(tag='Grad/E', scalar_value=avg_grad[0], global_step=epoch)
            logger.add_scalar(tag='Param/nu', scalar_value=nu, global_step=epoch)
            logger.add_scalar(tag='Grad/nu', scalar_value=avg_grad[1], global_step=epoch)
            logger.add_scalar(tag='Param/yield_stress', scalar_value=yield_stress, global_step=epoch)
            logger.add_scalar(tag='Grad/yield_stress', scalar_value=avg_grad[2], global_step=epoch)
            logger.add_scalar(tag='Param/rho', scalar_value=rho, global_step=epoch)
            logger.add_scalar(tag='Grad/rho', scalar_value=avg_grad[3], global_step=epoch)
            print(f"========> Epoch {epoch}: time={time() - t1}\n"
                  f"========> E={E}, nu={nu}, yield_stress={yield_stress}, rho={rho}")
            if arguments['param_set'] == 1:
                logger.add_scalar(tag='Param/manipulator_friction', scalar_value=manipulator_friction, global_step=epoch)
                logger.add_scalar(tag='Grad/manipulator_friction', scalar_value=avg_grad[4], global_step=epoch)
                logger.add_scalar(tag='Param/ground_friction', scalar_value=ground_friction, global_step=epoch)
                logger.add_scalar(tag='Grad/ground_friction', scalar_value=avg_grad[5], global_step=epoch)
                print(f"========> Epoch {epoch}: time={time() - t1}\n"
                      f"========> Manipulator friction={manipulator_friction}, ground friction={ground_friction}")

            """===========Validation==========="""
            validation_loss_config = loss_cfg.copy()
            validation_loss_config['exponential_distance'] = False
            print(f'=====> Run validation at epoch {epoch}')
            logging.info(f'=====> Run validation at epoch {epoch}')
            validation_loss = {
                'point_distance_sr': [],
                'point_distance_rs': [],
                'chamfer_loss_pcd': [],
                'particle_distance_sr': [],
                'particle_distance_rs': [],
                'chamfer_loss_particle': [],
                'height_map_loss_pcd': [],
                'emd_point_distance_loss': [],
                'emd_particle_distance_loss': [],
                'total_loss': []
            }
            dt_global = 0.01
            n_substeps = 50
            # training data_id_dict = {
            #     '1': {'rectangle': [3, 5], 'round': [0, 1], 'cylinder': [1, 2]},
            #     '2': {'rectangle': [1, 3], 'round': [0, 2], 'cylinder': [0, 2]},
            #     '3': {'rectangle': [1, 2], 'round': [0, 1], 'cylinder': [0, 4]},
            #     '4': {'rectangle': [1, 3], 'round': [1, 4], 'cylinder': [0, 4]},
            # }
            validation_dataind_dict = {
                '2': {
                    'rectangle': [4, 8],
                    'round': [4, 7],
                    'cylinder': [4, 7]
                },
                '4': {
                    'rectangle': [2, 4],
                    'round': [2, 3],
                    'cylinder': [1, 3]
                }
            }
            if arguments['param_set'] == 0:
                validation_motions = [2]
            else:
                validation_motions = [4]
            for validation_motion in validation_motions:
                for agent in agents:
                    agent_init_euler = (0, 0, 0)
                    if agent == 'rectangle':
                        agent_init_euler = (0, 0, 45)

                    trajectory = np.load(os.path.join(script_path, '..', 'trajectories', f'tr_{validation_motion}_v_dt_{dt_global:0.2f}.npy'))
                    validation_data_path = os.path.join(script_path, '..', f'data-motion-{validation_motion}', f'eef-{agent}')
                    data_inds = validation_dataind_dict[str(validation_motion)][agent]

                    horizon = trajectory.shape[0]
                    for data_ind in data_inds:
                        ti.reset()
                        ti.init(arch=backend, default_fp=DTYPE_TI, default_ip=ti.i32, fast_math=True, random_seed=seed,
                                debug=False, check_out_of_bound=False, device_memory_GB=3)
                        validation_data_cfg = {
                            'data_path': validation_data_path,
                            'data_ind': str(data_ind),
                        }
                        validation_env_cfg = {
                            'p_density': particle_density,
                            'horizon': horizon,
                            'dt_global': dt_global,
                            'n_substeps': n_substeps,
                            'material_id': 2,
                            'agent_name': agent,
                            'agent_init_euler': agent_init_euler,
                        }
                        env, mpm_env, init_state = make_env(validation_data_cfg, validation_env_cfg,
                                                            validation_loss_config, logger=logging)
                        set_parameters(mpm_env, validation_env_cfg['material_id'],
                                       E=E.copy(), nu=nu.copy(), yield_stress=yield_stress.copy(), rho=rho.copy(),
                                       ground_friction=ground_friction.copy(),
                                       manipulator_friction=manipulator_friction.copy())
                        loss_info = forward_backward(mpm_env, init_state, trajectory, backward=False)
                        # Check if the loss is strange
                        abort = False
                        particle_has_naninf = loss_info['particle_has_naninf']
                        if particle_has_naninf:
                            abort = True

                        if not abort:
                            for i, v in loss_info.items():
                                if i == 'final_height_map' or i == 'particle_has_naninf':
                                    pass
                                else:
                                    if np.isinf(v) or np.isnan(v):
                                        abort = True
                                        break

                        if abort:
                            print(f'===> [Warning] Aborting validation run: agent {agent}, data {data_ind}')
                            print(f'===> [Warning] Particle has nan or inf: {particle_has_naninf}')
                            print(f'===> [Warning] Strange loss.')
                            print(f'===> [Warning] E: {E}, nu: {nu}, yield stress: {yield_stress}')
                            print(f'===> [Warning] Rho: {rho}, ground friction: {ground_friction}, manipulator friction: {manipulator_friction}')
                            logging.error(f'===> [Warning] Aborting validation run: agent {agent}, data {data_ind}')
                            logging.error(f'===> [Warning] Particle has nan or inf: {particle_has_naninf}')
                            logging.error(f'===> [Warning] Strange loss.')
                            logging.error(f'===> [Warning] E: {E}, nu: {nu}, yield stress: {yield_stress}')
                            logging.error(f'===> [Warning] Rho: {rho}, ground friction: {ground_friction}, manipulator friction: {manipulator_friction}')
                            n_aborted_data += 1
                        else:
                            for i, v in validation_loss.items():
                                validation_loss[i].append(loss_info[i])

                        print(f'=====> Total loss: {mpm_env.loss.total_loss[None]}')
                        logging.info(f'=====> Total loss: {mpm_env.loss.total_loss[None]}')

                        mpm_env.simulator.clear_ckpt()

            for i, v in validation_loss.items():
                validation_loss[i] = np.mean(v)
            for i, v in validation_loss.items():
                print(f"========> Avg. Validation loss: {i}: {v}")
                logging.info(f"========> Avg. Validation loss: {i}: {v}")
                logger.add_scalar(tag=f'Validation loss/{i}', scalar_value=v, global_step=epoch)
            print(f"========> Num. aborted data so far: {n_aborted_data}")
            logging.info(f"========> Num. aborted data so far: {n_aborted_data}")

            logger.add_scalar(tag='Mem/GPU', scalar_value=get_gpu_memory()[0], global_step=epoch)
            logger.add_scalar(tag='Mem/RAM', scalar_value=process.memory_percent(), global_step=epoch)
            logger.add_scalar(tag='Aborted', scalar_value=n_aborted_data, global_step=epoch)

        logger.close()
        if arguments['param_set'] == 0:
            print(f"Final parameters: E={E}, nu={nu}, yield_stress={yield_stress}, rho={rho}")
            logging.info(f"Final parameters: E={E}, nu={nu}, yield_stress={yield_stress}, rho={rho}")
            np.save(os.path.join(log_dir, 'final_params.npy'), np.array([E, nu, yield_stress, rho], dtype=DTYPE_NP))
        else:
            print(f"Final parameters: E={E}, nu={nu}, yield_stress={yield_stress}, rho={rho}")
            print(f"Final parameters: manipulator friction={manipulator_friction}, ground friction={ground_friction}")
            logging.info(f"Final parameters: E={E}, nu={nu}, yield_stress={yield_stress}, rho={rho}")
            logging.info(f"Final parameters: manipulator friction={manipulator_friction}, ground friction={ground_friction}")
            np.save(os.path.join(log_dir, 'final_params.npy'), np.array([E, nu, yield_stress, rho, manipulator_friction, ground_friction], dtype=DTYPE_NP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_run', dest='n_run', type=int, default=-1)
    parser.add_argument('--seed', dest='seed', type=int, default=-1)
    parser.add_argument('--param_set', dest='param_set', type=int, default=0)
    parser.add_argument('--fewshot', dest='fewshot', default=False, action='store_true')
    parser.add_argument('--oneshot', dest='oneshot', default=False, action='store_true')
    parser.add_argument('--exp_dist', dest='exp_dist', default=False, action='store_true')
    parser.add_argument('--pd_rs_loss', dest='pd_rs_loss', default=False, action='store_true')
    parser.add_argument('--pd_sr_loss', dest='pd_sr_loss', default=False, action='store_true')
    parser.add_argument('--prd_rs_loss', dest='prd_rs_loss', default=False, action='store_true')
    parser.add_argument('--prd_sr_loss', dest='prd_sr_loss', default=False, action='store_true')
    parser.add_argument('--emd_p_loss', dest='emd_p_loss', default=False, action='store_true')
    parser.add_argument('--emd_pr_loss', dest='emd_pr_loss', default=False, action='store_true')
    parser.add_argument('--hm_loss', dest='hm_loss', default=False, action='store_true')
    parser.add_argument('--hm_res', dest='hm_res', default=32, type=int)
    parser.add_argument('--bs', dest='batchsize', default=20, type=int)
    parser.add_argument('--backend', dest='backend', default='opengl', type=str)
    args = vars(parser.parse_args())
    main(args)
