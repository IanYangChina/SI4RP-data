import numpy as np
import os
import taichi as ti
import matplotlib.pylab as plt
import matplotlib as mpl
from time import time
import argparse
import json
from doma.envs.sys_id_env import make_env
from doma.engine.utils.misc import set_parameters

DTYPE_NP = np.float32
DTYPE_TI = ti.f32
cuda_GB = 4
script_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_path, '..')

params = np.load(os.path.join(script_path, '..', 'optimisation-results',
                              f'level2-1cyl-run1-logs',
                              f'seed-1', 'final_params.npy')).flatten()
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
                              'eef-rectangle'),
    'data_ind': str(0),
}

env_cfg = {
    'p_density': 4e7,
    'horizon': 500,
    'dt_global': 0.01,
    'n_substeps': 50,
    'material_id': 2,
    'agent_name': 'cylinder',
    'agent_init_euler': (0, 0, 0),
}

cam_cfg = {
    'pos': (0.40, 0.1, 0.1),
    'lookat': (0.25, 0.25, 0.05),
    'fov': 30,
    'lights': [{'pos': (0.5, 0.25, 0.2), 'color': (0.6, 0.6, 0.6)},
               {'pos': (0.5, 0.5, 1.0), 'color': (0.6, 0.6, 0.6)},
               {'pos': (0.5, 0.0, 1.0), 'color': (0.8, 0.8, 0.8)}],
    'particle_radius': 0.002,
    'res': (640, 640),
    'euler': (135, 0, 180),
    'focal_length': 0.01
}


def reset_ti_and_env():
    ti.reset()
    ti.init(arch=ti.cuda, default_fp=DTYPE_TI, default_ip=ti.i32, fast_math=True, random_seed=0,
            debug=False, check_out_of_bound=False, device_memory_GB=cuda_GB)
    env, mpm_env, _ = make_env(data_cfg, env_cfg, loss_cfg, cam_cfg)
    mpm_env.agent.effectors[0].mesh.update_color((0.2, 0.2, 0.2, 1.0))
    set_parameters(mpm_env, env_cfg['material_id'],
                   e=E.copy(), nu=nu.copy(), yield_stress=yield_stress.copy(), rho=rho.copy(),
                   ground_friction=gf.copy(),
                   manipulator_friction=mf.copy())
    return env, mpm_env


def forward(mpm_env, init_state, init_agent_pos, trajectory,
            render=False, save_img=False, img_file_name=None):
    interval = trajectory.shape[0] // 10
    frames_to_save = [0, 2 * interval, 4 * interval, 6 * interval, 8 * interval, trajectory.shape[0] - 1]
    frames = []

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
        if save_img and (i in frames_to_save):
            img = mpm_env.render(mode='rgb_array')
            frames.append(img)

    loss_info = mpm_env.get_final_loss()

    if save_img:
        mpl.use('Agg')
        fig, axes = plt.subplots(1, len(frames_to_save), figsize=(len(frames_to_save) * 2, 2))
        plt.subplots_adjust(wspace=0.01)
        for i in range(len(frames_to_save)):
            img = frames[i]
            axes[i].imshow(img)
            axes[i].get_xaxis().set_visible(False)
            axes[i].get_yaxis().set_visible(False)
            axes[i].set_frame_on(False)
        if img_file_name is None:
            img_file_name = 'img_combine'
        plt.savefig(os.path.join(f'{img_file_name}_tr.pdf'), bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)

        plt.imshow(loss_info['final_height_map'], cmap='YlOrBr', vmin=0, vmax=60)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(f'{img_file_name}_hm.pdf'), bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

    return loss_info


def simulate_plan(init_agent_pos, init_state, actions, plan_ind):
    os.makedirs(os.path.join(script_path, '..', 'greedy-planning-logs', f'plan_{plan_ind}'), exist_ok=True)
    state = init_state
    for i in range(len(actions)):
        action = actions[i]
        if 'data_path' in action:
            continue
        env, mpm_env = reset_ti_and_env()
        tr_ind = action['action'][0]
        if tr_ind == 0:
            trajectory = np.zeros(shape=(10, 6))
        else:
            trajectory = np.load(os.path.join(script_path, '..', 'data',
                                              'trajectories', f'tr_poking-shifting_{tr_ind}_v_dt_0.01.npy'))
        agent_init_pos = init_agent_pos + np.array([action['action'][1] * 0.03, action['action'][2] * 0.03, 0])
        if action['action'][3] is not None:
            agent_init_pos[2] = action['action'][3] / 1000
        loss_info = forward(mpm_env, state, agent_init_pos, trajectory,
                            render=False, save_img=True,
                            img_file_name=os.path.join(script_path, '..', 'greedy-planning-logs', f'plan_{plan_ind}',
                                                       f'action_{i}'))
        print(f"Action {action}, Loss: {loss_info['height_map_loss_pcd']}")
        state = mpm_env.get_state()

    # fig, axes = plt.subplots(1, 2, figsize=(2 * 5, 5))
    # cmap = 'YlOrBr'
    # target_hm = mpm_env.loss.height_map_pcd_target.to_numpy()
    # min_val, max_val = np.amin(target_hm), np.amax(target_hm)
    # axes[0].imshow(target_hm, cmap=cmap, vmin=min_val, vmax=max_val)
    # axes[0].set_title('Target height map')
    # axes[1].imshow(loss_info['final_height_map'], cmap=cmap, vmin=min_val, vmax=max_val)
    # axes[1].set_title('Achieved height map')
    # plt.show()


def find_best_action(init_agent_pos, init_state, resultant_heightmap_z_max=None):
    best_action = (0, 0, 0)
    best_loss = np.inf
    finished_state = None
    resultant_heightmap_z_max_ = resultant_heightmap_z_max
    for action in [1, 2]:
        trajectory = np.load(os.path.join(script_path, '..', 'data',
                                          'trajectories', f'tr_poking-shifting_{action}_v_dt_0.01.npy'))
        for x in range(-1, 2):
            for y in range(-1, 2):
                t0 = time()
                env, mpm_env = reset_ti_and_env()
                agent_init_pos = init_agent_pos + np.array([x * 0.03, y * 0.03, 0])
                if resultant_heightmap_z_max is not None:
                    agent_init_pos[2] = resultant_heightmap_z_max / 1000
                print(f"Trying action {action}, offset {agent_init_pos}")
                # input("Press Enter to continue...")
                loss_info = forward(mpm_env, init_state, agent_init_pos, trajectory)
                print(f"Loss: {loss_info['height_map_loss_pcd']}")
                if loss_info['height_map_loss_pcd'] < best_loss:
                    best_loss = loss_info['height_map_loss_pcd']
                    best_action = (action, x, y, np.array([resultant_heightmap_z_max]).tolist()[0])
                    resultant_heightmap_z_max_ = np.max(loss_info['final_height_map'])
                    finished_state = mpm_env.get_state()
                    if resultant_heightmap_z_max_ < 0.02:
                        resultant_heightmap_z_max_ = 0.02
                print(f"Time taken: {time() - t0}")

    t0 = time()
    env, mpm_env = reset_ti_and_env()
    agent_init_pos = init_agent_pos
    if resultant_heightmap_z_max is not None:
        agent_init_pos[2] = resultant_heightmap_z_max / 1000
    print(f"Trying zero action")
    # input("Press Enter to continue...")
    loss_info = forward(mpm_env, init_state, agent_init_pos, np.zeros(shape=(10, 6)))
    print(f"Loss: {loss_info['height_map_loss_pcd']}")
    if loss_info['height_map_loss_pcd'] < best_loss:
        best_loss = loss_info['height_map_loss_pcd']
        best_action = (0, 0, 0, np.array([resultant_heightmap_z_max]).tolist()[0])
        resultant_heightmap_z_max_ = np.max(loss_info['final_height_map'])
        finished_state = mpm_env.get_state()
        if resultant_heightmap_z_max_ < 0.02:
            resultant_heightmap_z_max_ = 0.02
    print(f"Time taken: {time() - t0}")

    return best_action, best_loss, finished_state, resultant_heightmap_z_max_


def plot_target_height_map(colourbar_only=False):
    if not colourbar_only:
        plt.imshow(np.load(os.path.join(script_path, '..', 'data', 'data-planning-targets',
                                         'target_pcd_height_map-1-res32-vdsize0.001.npy')),
                    cmap='YlOrBr', vmin=0, vmax=60)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(script_path, '..', 'data', 'data-planning-targets',
                                 'target_pcd_height_map-1-res32-vdsize0.001.pdf'), bbox_inches='tight', pad_inches=0,
                    dpi=300)

        plt.close()
    else:
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure()
        ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])

        cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal',
                                       cmap='YlOrBr',
                                       norm=mpl.colors.Normalize(0, 60),
                                       ticks=[0, 20, 40, 60])

        plt.savefig('just_colorbar', bbox_inches='tight')
        plt.close()

def main(arguments):
    # Target height map ind
    # loss_cfg['target_ind'] = 0
    # Object initial configuration
    # data_cfg['data_path'] = os.path.join(script_path, '..', 'data', 'data-motion-long-horizon',
    #                                      'eef-rectangle')
    # data_cfg['data_ind'] = str(0)

    # Target height map ind
    loss_cfg['target_ind'] = arguments['target_id']
    # Object initial configuration
    data_cfg['data_path'] = os.path.join(script_path, '..', 'data', 'data-motion-poking-shifting-1',
                                         'eef-rectangle')
    data_cfg['data_ind'] = str(1)

    evaluate = arguments['evaluate']
    log_p_dir = os.path.join(script_path, '..', 'greedy-planning-logs')
    os.makedirs(log_p_dir, exist_ok=True)

    if evaluate:
        n = 1
        actions = json.load(open(os.path.join(log_p_dir, f'actions_{n}.json')))
        data_cfg['data_path'] = actions[-1]['data_path']
        data_cfg['data_ind'] = actions[-1]['data_ind']
        env, mpm_env = reset_ti_and_env()
        init_state = mpm_env.get_state()
        init_agent_pos = np.asarray(mpm_env.agent.effectors[0].init_pos)

        simulate_plan(init_agent_pos, init_state, actions, n)
    else:
        env, mpm_env = reset_ti_and_env()
        init_state = mpm_env.get_state()
        init_agent_pos = np.asarray(mpm_env.agent.effectors[0].init_pos)

        actions = []
        hm_z_max = None
        for n in range(10):
            best_action, best_loss, new_state, hm_z_max = find_best_action(init_agent_pos,
                                                                           init_state,
                                                                           hm_z_max)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', dest='evaluate', action='store_true', default=False)
    parser.add_argument('--target-id', dest='target_id', type=int, default=1)
    args = parser.parse_args()
    main(args)
