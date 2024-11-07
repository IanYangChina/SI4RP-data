import numpy as np
import os
import taichi as ti
import matplotlib.pylab as plt

import json
from doma.envs.sys_id_env import make_env
from doma.engine.utils.misc import get_gpu_memory, set_parameters
DTYPE_NP = np.float32
DTYPE_TI = ti.f32
script_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_path, '..')


def forward(mpm_env, init_state, trajectory):
    # Forward
    mpm_env.set_state(init_state['state'], grad_enabled=True)
    for i in range(mpm_env.horizon):
        action = trajectory[i]
        mpm_env.step(action)

    loss_info = mpm_env.get_final_loss()
    return loss_info


def find_best_action(loss_config, env_config, trajectory):
    for action in range(2):
        for x in range(-2, 2):
            for y in range(-2, 2):
                agent_init_pos_xy_offset = (x*0.015, y*0.015)


def main():
    loss_cfg = {
        'exponential_distance': False,
        'averaging_loss': False,
        'point_distance_rs_loss': False,
        'point_distance_sr_loss': False,
        'particle_distance_rs_loss': False,
        'particle_distance_sr_loss': False,
        'emd_point_distance_loss': False,
        'emd_particle_distance_loss': False,
        'height_map_loss': False,
        'down_sample_voxel_size': 0.005,
        'voxelise_res': 1080,
        'ptcl_density': 4e6,
        'load_height_map': True,
        'height_map_res': 32,
        'height_map_size': 0.11,
    }
    data_cfg = {
        'data_path': os.path.join(script_path, '..', 'data', 'data-motion-long-horizon',
                                  'eef-cylinder'),
        'data_ind': str(0),
    }
    trajectory = np.load(os.path.join(script_path, '..', 'data',
                                      'trajectories', f'tr_poking_shifting_2_v_dt_0.01.npy'))

    env_cfg = {
        'p_density': 4e6,
        'horizon': trajectory.shape[0],
        'dt_global': 0.01,
        'n_substeps': 50,
        'material_id': 2,
        'agent_name': 'cylinder',
        'agent_init_euler': agent_init_euler,
    }

    ti.reset()
    ti.init(arch=ti.cuda, default_fp=DTYPE_TI, default_ip=ti.i32, fast_math=True, random_seed=0,
            debug=False, check_out_of_bound=False, device_memory_GB=3)

    env, mpm_env, init_state = make_env(data_cfg, env_cfg, loss_cfg, logger=logging)
    set_parameters(mpm_env, env_cfg['material_id'],
                   e=E.copy(), nu=nu.copy(), yield_stress=yield_stress.copy(), rho=rho.copy(),
                   ground_friction=ground_friction.copy(),
                   manipulator_friction=manipulator_friction.copy())
