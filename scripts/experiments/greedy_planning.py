import numpy as np
import os
import taichi as ti
import matplotlib.pylab as plt
from time import time

import json
from doma.envs.sys_id_env import make_env
from doma.engine.utils.misc import set_parameters
DTYPE_NP = np.float32
DTYPE_TI = ti.f32
cuda_GB = 5
script_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_path, '..')

data_dir = os.path.join(script_path, '..', 'optimisation-results',
                        f'level2-1cyl-run1-logs',
                        f'seed-1')
params = np.load(os.path.join(data_dir, 'final_params.npy')).flatten()
E = params[0]
nu = params[1]
yield_stress = params[2]
rho = params[3]
mf = params[4]
gf = params[5]

loss_cfg = {
    'planning': True,
    'target_ind': 0,
    'height_map_res': 32,
    'height_map_size': 0.11,
}

data_cfg = {
    'data_path': os.path.join(script_path, '..', 'data', 'data-motion-long-horizon',
                              'eef-cylinder'),
    'data_ind': str(0),
}

env_cfg = {
    'p_density': 4e6,
    'horizon': 500,
    'dt_global': 0.01,
    'n_substeps': 50,
    'material_id': 2,
    'agent_name': 'cylinder',
    'agent_init_euler': (0, 0, 0),
}


def reset_ti_and_env():
    ti.reset()
    ti.init(arch=ti.cuda, default_fp=DTYPE_TI, default_ip=ti.i32, fast_math=True, random_seed=0,
            debug=False, check_out_of_bound=False, device_memory_GB=cuda_GB)
    env, mpm_env, _ = make_env(data_cfg, env_cfg, loss_cfg)
    set_parameters(mpm_env, env_cfg['material_id'],
                   e=E.copy(), nu=nu.copy(), yield_stress=yield_stress.copy(), rho=rho.copy(),
                   ground_friction=gf.copy(),
                   manipulator_friction=mf.copy())
    return env, mpm_env


def forward(mpm_env, init_state, init_agent_pos, trajectory, render=False):
    # Forward
    mpm_env.set_state(init_state['state'], grad_enabled=False)
    init_agent_p = np.append(init_agent_pos, mpm_env.agent.effectors[0].init_rot)
    mpm_env.agent.effectors[0].set_state(0, init_agent_p)
    if render:
        mpm_env.render("human")
    for i in range(trajectory.shape[0]):
        action = trajectory[i]
        mpm_env.step(action)
        if render:
            mpm_env.render("human")

    loss_info = mpm_env.get_final_loss()
    return loss_info


def simulate_plan(init_agent_pos, init_state, actions):
    state = init_state
    for action in actions:
        if 'data_path' in action:
            continue
        env, mpm_env = reset_ti_and_env()
        tr_ind = action['action'][0]
        trajectory = np.load(os.path.join(script_path, '..', 'data',
                                          'trajectories', f'tr_poking-shifting_{tr_ind}_v_dt_0.01.npy'))
        agent_init_pos = init_agent_pos + np.array([action['action'][1] * 0.015, action['action'][2] * 0.015, 0])
        loss_info = forward(mpm_env, state, agent_init_pos, trajectory, render=True)
        print(f"Action {action}, Loss: {loss_info['height_map_loss_pcd']}")
        state = mpm_env.get_state()

    fig, axes = plt.subplots(1, 2, figsize=(2 * 5, 5))
    cmap = 'YlOrBr'
    target_hm = mpm_env.loss.height_map_pcd_target.to_numpy()
    min_val, max_val = np.amin(target_hm), np.amax(target_hm)
    axes[0].imshow(target_hm, cmap=cmap, vmin=min_val, vmax=max_val)
    axes[0].set_title('Target height map')
    axes[1].imshow(loss_info['final_height_map'], cmap=cmap, vmin=min_val, vmax=max_val)
    axes[1].set_title('Achieved height map')
    plt.show()


def find_best_action(init_agent_pos, init_state):
    best_action = (0, 0, 0)
    best_loss = np.inf
    finished_state = None
    for action in [1, 2]:
        trajectory = np.load(os.path.join(script_path, '..', 'data',
                                          'trajectories', f'tr_poking-shifting_{action}_v_dt_0.01.npy'))
        for x in range(-1, 1):
            for y in range(-1, 1):
                t0 = time()
                env, mpm_env = reset_ti_and_env()
                agent_init_pos = init_agent_pos + np.array([x*0.015, y*0.015, 0])
                print(f"Trying action {action}, offset {agent_init_pos}")
                # input("Press Enter to continue...")
                loss_info = forward(mpm_env, init_state, agent_init_pos, trajectory)
                print(f"Loss: {loss_info['height_map_loss_pcd']}")
                if loss_info['height_map_loss_pcd'] < best_loss:
                    best_loss = loss_info['height_map_loss_pcd']
                    best_action = (action, x, y)
                    finished_state = mpm_env.get_state()
                    resultant_heightmap_z_max = np.max(loss_info['final_height_map'])
                print(f"Time taken: {time() - t0}")

    return best_action, best_loss, finished_state


def main():
    # Target height map ind
    loss_cfg['target_ind'] = 0
    # Object initial configuration
    data_cfg['data_path'] = os.path.join(script_path, '..', 'data', 'data-motion-long-horizon',
                                         'eef-cylinder')
    data_cfg['data_ind'] = str(0)

    evaluate = True
    log_p_dir = os.path.join(script_path, '..', 'greedy-planning-logs')
    os.makedirs(log_p_dir, exist_ok=True)

    env, mpm_env = reset_ti_and_env()
    init_state = mpm_env.get_state()
    init_agent_pos = np.asarray(mpm_env.agent.effectors[0].init_pos)

    if evaluate:
        n = 0
        actions = json.load(open(os.path.join(log_p_dir, f'actions_{n}.json')))
        simulate_plan(init_agent_pos, init_state, actions)
    else:
        actions = []
        for n in range(5):
            best_action, best_loss, new_state = find_best_action(init_agent_pos, init_state)
            print(f"Best action: {best_action}, Best loss: {best_loss}")
            actions.append({
                'action': best_action,
                'loss': best_loss
            })
            init_state = new_state

        actions.append(data_cfg)
        n = 0
        done = False
        while not done:
            file_name = os.path.join(log_p_dir, f'actions_{n}.json')
            if not os.path.exists(file_name):
                with open(os.path.join(log_p_dir, f'actions_{n}.json'), 'w') as f:
                    json.dump(actions, f, indent=4)
                done = True
            else:
                n += 1


if __name__ == '__main__':
    main()
