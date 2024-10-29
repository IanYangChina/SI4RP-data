import os
import argparse
import taichi as ti
import numpy as np
from time import time
from torch.utils.tensorboard import SummaryWriter
from doma.optimiser.adam import Adam, GD
from doma.envs.sys_id_env import make_env
from doma.engine.utils.misc import get_gpu_memory, set_parameters
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
    script_path = os.path.join(script_path, '..')
    result_path = os.path.join(script_path, '..', 'optimisation-results')

    DTYPE_NP = np.float32
    DTYPE_TI = ti.f32
    dt_global = 0.01
    particle_density = arguments['ptcl_density']
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
        'emd_point_distance_loss': arguments['emd_p_loss'],
        'emd_particle_distance_loss': arguments['emd_pr_loss'],
        'height_map_loss': arguments['hm_loss'],
        'down_sample_voxel_size': 0.005,
        'voxelise_res': 1080,
        'ptcl_density': particle_density,
        'load_height_map': True,
        'height_map_res': 32,
        'height_map_size': 0.11,
    }
    if arguments['soil']:
        loss_cfg['height_map_size'] = 0.17
    if arguments['slime']:
        loss_cfg['height_map_size'] = 0.15

    # Parameter ranges
    E_range = (1e4, 3e5)
    nu_range = (0.01, 0.48)
    yield_stress_range = (1e3, 2e4)
    rho_range = (1000, 2000)
    mf_range = (0.01, 2.0)
    gf_range = (0.01, 2.0)

    agents = ['rectangle', 'round', 'cylinder']
    contact_level = arguments['contact_level']
    assert contact_level in [1, 2], 'contact_level must be 1 or 2'
    dataset = arguments['dataset']
    assert dataset in ['12mix', '6mix', '1cyl', '1rec', '1round'], 'dataset must be 12mix, 6mix, 1cyl, 1rec, or 1round'
    if arguments['slime']:
        assert not arguments['soil'], 'Cannot use both slime and soil datasets.'
        contact_level = 2
        dataset = 'slime'
    if arguments['soil']:
        assert not arguments['slime'], 'Cannot use both slime and soil datasets.'
        contact_level = 2
        dataset = 'soil'
    motion_name = 'poking' if contact_level == 1 else 'poking-shifting'

    n_epoch = 100
    n_aborted_data = 0
    if arguments['seed'] != -1:
        seeds = [arguments['seed']]
    else:
        seeds = [0, 1, 2]
    if arguments['n_run'] != -1:
        # Append experiment into existing folder
        n = arguments['n_run']
        log_p_dir = os.path.join(result_path, f'level{contact_level}-{dataset}-run{n}-logs')
    else:
        # New experiments
        n = 0
        while True:
            log_p_dir = os.path.join(result_path, f'level{contact_level}-{dataset}-run{n}-logs')
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
    logging.info(f"=====> Logging to {log_file_name}")

    training_config = {
        'contact_level': contact_level,
        'lr_E': 4e3,
        'lr_nu': 1e-2,
        'lr_yield_stress': 5e2,
        'lr_rho': 10,
        'lr_manipulator_friction': 1e-2,
        'lr_ground_friction': 1e-2,
        'n_epoch': n_epoch,
        'seeds': seeds,
    }
    if contact_level == 2:
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

    print(f"=====> Optimisation at contact_level {contact_level} with dataset {dataset} for {n_epoch} epochs for {len(seeds)} random seeds.")
    print(f"=====> Loss config: {loss_cfg}")
    print(f"=====> Training config: {training_config}")
    logging.info(f"=====> Optimisation at contact_level {contact_level} with dataset {dataset} for {n_epoch} epochs for {len(seeds)} random seeds.")
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
        if contact_level == 2:
            manipulator_friction = np.asarray(np.random.uniform(mf_range[0], mf_range[1]), dtype=DTYPE_NP).reshape((1,))  # Manipulator friction
            ground_friction = np.asarray(np.random.uniform(gf_range[0], gf_range[1]), dtype=DTYPE_NP).reshape((1,))

            print(f"=====> Seed: {seed}, initial parameters: manipulation friction={manipulator_friction}, ground friction={ground_friction}")
            logging.info(f"=====> Seed: {seed}, initial parameters: manipulation friction={manipulator_friction}, ground friction={ground_friction}")
            # Optimiser: Adam
            optim_mf = Adam(parameters_shape=manipulator_friction.shape,
                            cfg={'lr': training_config['lr_manipulator_friction'], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
            optim_gf = Adam(parameters_shape=ground_friction.shape,
                            cfg={'lr': training_config['lr_ground_friction'], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})

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

            # datasets
            if arguments['dataset'] == '12mix':
                n_datapoints = 12
                motion_ids = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
                agent_ids = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
                data_inds = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            elif arguments['dataset'] == '6mix':
                n_datapoints = 6
                motion_ids = [1, 1, 1, 2, 2, 2]
                agent_ids = [0, 1, 2, 0, 1, 2]
                data_inds = [0, 0, 0, 0, 0, 0]
            else:
                n_datapoints = 1
                motion_ids = [2]
                data_inds = [0]

                if arguments['dataset'] == '1rec':
                    agent_ids = [0]
                elif arguments['dataset'] == '1round':
                    agent_ids = [1]
                else:
                    agent_ids = [2]

            if arguments['slime'] or arguments['soil']:
                n_datapoints = 1
                motion_ids = [2]
                agent_ids = [2]
                data_inds = [0]

            for i in range(n_datapoints):
                motion_id = str(motion_ids[i])
                trajectory = np.load(os.path.join(script_path, '..', 'data', 'trajectories',
                                                  f'tr_{motion_name}_{motion_id}_v_dt_{dt_global:0.2f}.npy'))
                horizon = trajectory.shape[0]

                agent = agents[agent_ids[i]]
                agent_init_euler = (0, 0, 0)
                if agent == 'rectangle':
                    agent_init_euler = (0, 0, 45)
                training_data_path = os.path.join(script_path, '..', 'data',
                                                  f'data-motion-{motion_name}-{motion_id}', f'eef-{agent}')
                if arguments['slime']:
                    training_data_path = os.path.join(script_path, '..', 'data', 'other_mats', 'slime')
                if arguments['soil']:
                    training_data_path = os.path.join(script_path, '..', 'data', 'other_mats', 'soil')

                data_ind = data_inds[i]

                ti.reset()
                ti.init(arch=backend, default_fp=DTYPE_TI, default_ip=ti.i32, fast_math=True, random_seed=seed,
                        debug=False, check_out_of_bound=False, device_memory_GB=arguments['device_memory_GB'])
                data_cfg = {
                    'data_path': training_data_path,
                    'data_ind': str(data_ind),
                }
                env_cfg = {
                    'p_density': particle_density,
                    'horizon': horizon,
                    'dt_global': dt_global,
                    'n_substeps': 50,
                    'material_id': 2,
                    'agent_name': agent,
                    'agent_init_euler': agent_init_euler,
                }
                print(f'=====> Computing: epoch {epoch}, motion {motion_name}-{motion_id}, agent {agent}, data {data_ind}')
                logging.info(f'=====> Computing: epoch {epoch}, motion {motion_name}-{motion_id}, agent {agent}, data {data_ind}')
                env, mpm_env, init_state = make_env(data_cfg, env_cfg, loss_cfg, logger=logging)
                set_parameters(mpm_env, env_cfg['material_id'],
                               e=E.copy(), nu=nu.copy(), yield_stress=yield_stress.copy(), rho=rho.copy(),
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
                    print(f'===> [Warning] Aborting datapoint: epoch {epoch}, motion {motion_name}-{motion_id}, agent {agent}, data {data_ind}')
                    print(f'===> [Warning] Particle has nan or inf: {particle_has_naninf}')
                    print(f'===> [Warning] Strange loss or gradient.')
                    print(f'===> [Warning] E: {E}, nu: {nu}, yield stress: {yield_stress}')
                    print(f'===> [Warning] Rho: {rho}, ground friction: {ground_friction}, manipulator friction: {manipulator_friction}')
                    logging.error(f'===> [Warning] Aborting datapoint: epoch {epoch}, motion {motion_name}-{motion_id}, agent {agent}, data {data_ind}')
                    logging.error(f'===> [Warning] Particle has nan or inf: {particle_has_naninf}')
                    logging.error(f'===> [Warning] Strange loss or gradient.')
                    logging.error(f'===> [Warning] E: {E}, nu: {nu}, yield stress: {yield_stress}')
                    logging.error(f'===> [Warning] Rho: {rho}, ground friction: {ground_friction}, manipulator friction: {manipulator_friction}')
                    n_aborted_data += 1
                else:
                    for j, v in loss.items():
                        loss[j].append(loss_info[j])
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
            if arguments['hm_loss']:
                avg_grad *= -1.0

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
            if contact_level == 2:
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
            print(f"========> Epoch {epoch}: time={time() - t1} seconds\n"
                  f"========> E={E}, nu={nu}, yield_stress={yield_stress}, rho={rho}")
            if contact_level == 2:
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
            if arguments['slime'] or arguments['soil']:
                agents_to_validate = ['cylinder']
            else:
                agents_to_validate = agents
            for agent in agents_to_validate:
                agent_init_euler = (0, 0, 0)
                if agent == 'rectangle':
                    agent_init_euler = (0, 0, 45)

                trajectory = np.load(os.path.join(script_path, '..', 'data',
                                                  'trajectories', f'tr_{motion_name}_2_v_dt_{dt_global:0.2f}.npy'))
                validation_data_path = os.path.join(script_path, '..', 'data',
                                                    f'data-motion-{motion_name}-2', f'eef-{agent}', 'validation_data')
                data_inds = [0, 1]
                if arguments['slime']:
                    data_inds = [0]
                    trajectory = np.load(os.path.join(script_path, '..', 'data', 'trajectories', f'tr_long-horizon-{agent}_v_dt_{dt_global:0.2f}.npy'))
                    validation_data_path = os.path.join(script_path, '..', 'data', 'other_mats', 'slime', 'long-motion-validation')
                if arguments['soil']:
                    data_inds = [0]
                    trajectory = np.load(os.path.join(script_path, '..', 'data', 'trajectories', f'tr_long-horizon-{agent}_v_dt_{dt_global:0.2f}.npy'))
                    validation_data_path = os.path.join(script_path, '..', 'data', 'other_mats', 'soil', 'long-motion-validation')
                horizon = trajectory.shape[0]
                for data_ind in data_inds:
                    ti.reset()
                    ti.init(arch=backend, default_fp=DTYPE_TI, default_ip=ti.i32, fast_math=True, random_seed=seed,
                            debug=False, check_out_of_bound=False, device_memory_GB=arguments['device_memory_GB'])
                    validation_data_cfg = {
                        'data_path': validation_data_path,
                        'data_ind': str(data_ind),
                    }
                    validation_env_cfg = {
                        'p_density': particle_density,
                        'horizon': horizon,
                        'dt_global': dt_global,
                        'n_substeps': 50,
                        'material_id': 2,
                        'agent_name': agent,
                        'agent_init_euler': agent_init_euler,
                    }
                    env, mpm_env, init_state = make_env(validation_data_cfg, validation_env_cfg,
                                                        validation_loss_config, logger=logging)
                    set_parameters(mpm_env, validation_env_cfg['material_id'],
                                   e=E.copy(), nu=nu.copy(), yield_stress=yield_stress.copy(), rho=rho.copy(),
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
        if contact_level == 1:
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
    parser.add_argument('--n_run', dest='n_run', type=int, default=-1, help='Run number. If -1, create a new run, otherwise store into the given run folder.')
    parser.add_argument('--seed', dest='seed', type=int, default=-1, help='Random seed. If -1, use 0, 1, 2.')
    parser.add_argument('--slime', dest='slime', default=False, action='store_true', help='Use slime dataset.')
    parser.add_argument('--soil', dest='soil', default=False, action='store_true', help='Use soil dataset.')
    parser.add_argument('--con_lv', dest='contact_level', type=int, default=1, choices=[1, 2], help='Contact level: 1 or 2')
    parser.add_argument('--dataset', dest='dataset', type=str, default='12mix', choices=['12mix', '6mix', '1cyl', '1rec', '1round'], help='Dataset: 12mix, 6mix, 1cyl, 1rec, 1round')
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=float, default=4e7, help='Particle density')
    parser.add_argument('--cd_p_loss', dest='cd_point_distance_loss', default=False, action='store_true', help='Count Chamfer loss between real points and simulated particles into loss computation.')
    parser.add_argument('--cd_pr_loss', dest='cd_particle_distance_loss', default=False, action='store_true', help='Count Chamfer loss between reconstructed particles and simulated particles into loss computation.')
    parser.add_argument('--emd_p_loss', dest='emd_p_loss', default=False, action='store_true', help='Count EMD loss between real points and simulated particles into loss computation.')
    parser.add_argument('--emd_pr_loss', dest='emd_pr_loss', default=False, action='store_true', help='Count EMD loss between reconstructed particles and simulated particles into loss computation.')
    parser.add_argument('--hm_loss', dest='hm_loss', default=False, action='store_true', help='Count height map loss into loss computation.')
    parser.add_argument('--backend', dest='backend', default='cuda', type=str, choices=['opengl', 'cuda', 'vulkan'], help='Computation backend: opengl, cuda, vulkan')
    parser.add_argument('--device_mem', dest='device_memory_GB', default=5, type=int, help='Device memory in GB, depending on your GPU device, if out of memory, increase this value.')
    args = vars(parser.parse_args())
    main(args)
