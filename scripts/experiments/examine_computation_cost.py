import os
import argparse
import taichi as ti
import numpy as np
from time import time
from doma.envs.sys_id_env import make_env
from doma.engine.utils.misc import get_gpu_memory, set_parameters
import psutil

MATERIAL_ID = 2
process = psutil.Process(os.getpid())
script_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_path, '..')


def forward_backward(mpm_env, init_state, trajectory, backward=False):
    # Forward
    t0 = time()
    mpm_env.set_state(init_state['state'], grad_enabled=True)
    for i in range(mpm_env.horizon):
        action = trajectory[i]
        mpm_env.step(action)

    loss_info = mpm_env.get_final_loss()

    print(f'===> Forward time: {time() - t0:.2f} s')
    t1 = time()

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
        print(f'===> Backward time: {time() - t1:.2f} s')
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

    DTYPE_NP = np.float32
    DTYPE_TI = ti.f32
    dt_global = 0.01
    particle_density = arguments['ptcl_density']
    loss_cfg = {
        'exponential_distance': False,
        'averaging_loss': False,
        'point_distance_rs_loss': False,
        'point_distance_sr_loss': False,
        'particle_distance_rs_loss': False,
        'particle_distance_sr_loss': False,
        'emd_point_distance_loss': False,
        'emd_particle_distance_loss': True,
        'height_map_loss': False,
        'down_sample_voxel_size': 0.005,
        'voxelise_res': 1080,
        'ptcl_density': particle_density,
        'load_height_map': True,
        'height_map_res': 32,
        'height_map_size': 0.11,
    }
    run_id = 3
    seed_id = 0
    dataset = '1round'
    contact_level = 2
    data_dir = os.path.join(script_path, '..', 'optimisation-results',
                            f'level{contact_level}-{dataset}-run{run_id}-logs',
                            f'seed-{seed_id}')
    params = np.load(os.path.join(data_dir, 'final_params.npy')).flatten()
    E = params[0]
    nu = params[1]
    yield_stress = params[2]
    rho = params[3]
    mf = params[4]
    gf = params[5]

    agent = 'cylinder'
    if arguments['long_motion']:
        data_path = os.path.join(script_path, '..', 'data', 'data-motion-long-horizon', f'eef-{agent}')
        trajectory = np.load(os.path.join(script_path, '..', 'data', 'trajectories', f'tr_long-horizon-{agent}_v_dt_{dt_global:0.2f}.npy'))
    else:
        if arguments['contact_level'] == 1:
            motion_name = 'poking'
        else:
            motion_name = 'poking-shifting'
        motion_ind = arguments['motion_id']
        data_path = os.path.join(script_path, '..', 'data', f'data-motion-{motion_name}-{motion_ind}', f'eef-{agent}')
        trajectory = np.load(os.path.join(script_path, '..', 'data', 'trajectories', f'tr_{motion_name}_{motion_ind}_v_dt_{dt_global:0.2f}.npy'))

    horizon = trajectory.shape[0]

    ti.reset()
    ti.init(arch=backend, default_fp=DTYPE_TI, default_ip=ti.i32, debug=False, device_memory_GB=arguments['cuda_GB'],
            fast_math=True)
    data_cfg = {
        'data_path': data_path,
        'data_ind': str(0),
    }
    env_cfg = {
        'p_density': particle_density,
        'horizon': horizon,
        'dt_global': dt_global,
        'n_substeps': 50,
        'material_id': 2,
        'agent_name': agent,
        'agent_init_euler': (0, 0, 0),
    }
    # print(f'===> CPU memory occupied before create env: {process.memory_percent()} %')
    # print(f'===> GPU memory before create env: {get_gpu_memory()}')
    env, mpm_env, init_state = make_env(data_cfg, env_cfg, loss_cfg)
    mpm_env.agent.effectors[0].mesh.update_color((0.2, 0.2, 0.2, 1.0))
    print(f'===> Num. simulation particles: {mpm_env.loss.n_particles_matching_mat}')
    print(f'===> Num. target pcd points: {mpm_env.loss.n_target_pcd_points}')
    print(f'===> Num. target particles: {mpm_env.loss.n_target_particles_from_mesh}')
    # print(f'===> CPU memory occupied after create env: {process.memory_percent()} %')
    # print(f'===> GPU memory after create env: {get_gpu_memory()}')

    print(f'===> Parameters: E = {E}, nu = {nu}, yield_stress = {yield_stress}, rho = {rho}, gf = {gf}, mf = {mf}')
    set_parameters(mpm_env, env_cfg['material_id'],
                   e=E, nu=nu, yield_stress=yield_stress,
                   rho=rho, ground_friction=gf, manipulator_friction=mf)
    print('First run')
    forward_backward(mpm_env, init_state, trajectory.copy(), backward=arguments['backward'])
    print('Second run')
    forward_backward(mpm_env, init_state, trajectory.copy(), backward=arguments['backward'])
    print('Third run')
    forward_backward(mpm_env, init_state, trajectory.copy(), backward=arguments['backward'])

    # print(f'===> CPU memory occupied after forward: {process.memory_percent()} %')
    # print(f'===> GPU memory after forward: {get_gpu_memory()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', dest='backend', default='cuda', type=str, choices=['opengl', 'cuda', 'vulkan'], help='Computation backend: opengl, cuda, vulkan')
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=float, default=4e7, help='Particle density')
    parser.add_argument('--con_lv', dest='contact_level', type=int, default=1, help='Contact level')
    parser.add_argument('--m_id', dest='motion_id', type=int, default=1, help='Motion ID')
    parser.add_argument('--long_motion', dest='long_motion', default=False, action='store_true', help='Examine long horizon motion simulation. This diseffects the contact_level and motion_ind arguments.')
    parser.add_argument('--backward', dest='backward', default=False, action='store_true')
    parser.add_argument('--cuda_GB', dest='cuda_GB', type=float, default=3, help='GPU memory in GB')
    args = parser.parse_args()
    main(vars(args))
