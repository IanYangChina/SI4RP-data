import os
import numpy as np
import open3d as o3d
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import taichi as ti
from time import time, sleep

script_path = os.path.dirname(os.path.realpath(__file__))
# Load parameters
params = np.load(os.path.join(script_path, '..', 'data-motion-1', 'optimisation-logs-vanilla-chamfer', 'seed-2', 'final_params.npy'))
print(f'Initial parameters e, nu, ys: {params}')
e = params[0]  # Young's modulus
nu = params[1]  # Poisson's ratio
yield_stress = params[2]

# Trajectory 1 presses down 0.015 m and lifts for 0.03 m
# In simulation we only takes the pressing down part
real_horizon_1 = int(0.03 / 0.001)
v = 0.045 / 0.03  # 1.5 m/s
horizon_1 = int((0.015 / v) / 0.001)  # 5 steps
trajectory_1 = np.zeros(shape=(horizon_1, 6))
trajectory_1[:, 2] = -v

# Trajectory 2 presses down 0.02 m and lifts for 0.03 m
# In simulation we only takes the pressing down part
real_horizon_2 = int(0.04 / 0.001)
v = 0.05 / 0.04  # 1.25 m/s
horizon_2 = int((0.02 / v) / 0.001)  # 8 steps
trajectory_2 = np.zeros(shape=(horizon_2, 6))
trajectory_2[:, 2] = -v

agent_1 = 'rectangle'
agent_2 = 'round'

horizon = horizon_1
trajectory = trajectory_1

d_pcd_sr_list = []
d_pcd_rs_list = []
d_pcd_total_list = []
d_particle_sr_list = []
d_particle_rs_list = []
d_particle_total_list = []
for agent in [agent_1, agent_2]:
    # Loading mesh
    data_path = os.path.join(script_path, '..', 'data-motion-1', f'eef-{agent}')
    for data_ind in range(9):
        mesh_file_path = os.path.join(data_path, 'mesh_' + str(data_ind)+str(0) + '_repaired_normalised.obj')
        centre_real = np.load(os.path.join(data_path, 'mesh_' + str(data_ind)+str(0) + '_repaired_centre.npy'))
        centre_top_normalised = np.load(os.path.join(data_path, 'mesh_' + str(data_ind)+str(0) + '_repaired_normalised_centre_top.npy'))
        centre_top_normalised_ = np.load(os.path.join(data_path, 'mesh_' + str(data_ind)+str(1) + '_repaired_normalised_centre_top.npy'))

        # Building environment
        initial_pos = (0.25, 0.25, centre_top_normalised[-1] + 0.01)
        agent_init_pos = (0.25, 0.25, 2*centre_top_normalised[-1]+0.01)
        material_id = 2

        ti.init(arch=ti.vulkan, device_memory_GB=5, default_fp=ti.f32, fast_math=False)
        from doma.envs import SysIDEnv
        env = SysIDEnv(ptcl_density=3e7, horizon=horizon,
                       mesh_file=mesh_file_path, material_id=material_id, voxelise_res=1080, initial_pos=initial_pos,
                       target_pcd_file=os.path.join(data_path, 'pcd_' + str(data_ind)+str(1) + '.ply'),
                       pcd_offset=(-centre_real + initial_pos), mesh_offset=(0.25, 0.25, centre_top_normalised_[-1] + 0.01),
                       target_mesh_file=os.path.join(data_path, 'mesh_' + str(data_ind)+str(1) + '_repaired_normalised.obj'),
                       loss_weight=1.0, separate_param_grad=False,
                       render_agent=True,
                       agent_cfg_file=agent+'_eef.yaml', agent_init_pos=agent_init_pos, agent_init_euler=(0, 0, 45))
        mpm_env = env.mpm_env

        # Initialising parameters
        mpm_env.simulator.system_param[None].yield_stress = np.array(yield_stress, dtype=np.float32)
        mpm_env.simulator.particle_param[material_id].E = np.array(e, dtype=np.float32)
        mpm_env.simulator.particle_param[material_id].nu = np.array(nu, dtype=np.float32)
        env.reset()

        d_pcd_sr = np.zeros(shape=(horizon,))
        d_pcd_rs = np.zeros(shape=(horizon,))
        d_pcd_total = np.zeros(shape=(horizon,))
        d_particle_sr = np.zeros(shape=(horizon,))
        d_particle_rs = np.zeros(shape=(horizon,))
        d_particle_total = np.zeros(shape=(horizon,))

        init_state = mpm_env.get_state()
        # Forward
        t1 = time()
        mpm_env.set_state(init_state['state'], grad_enabled=False)
        for i in range(mpm_env.horizon):
            action = trajectory[i]
            mpm_env.step(action)
            # env.render(mode='human')
            step_loss_info = mpm_env.get_step_loss()
            d_pcd_sr[i] = step_loss_info['avg_point_distance_sr']
            d_pcd_rs[i] = step_loss_info['avg_point_distance_rs']
            d_pcd_total[i] = step_loss_info['chamfer_loss_pcd']
            d_particle_sr[i] = step_loss_info['avg_particle_distance_sr']
            d_particle_rs[i] = step_loss_info['avg_particle_distance_rs']
            d_particle_total[i] = step_loss_info['chamfer_loss_particle']
            # sleep(0.5)

        final_loss_info = mpm_env.get_final_loss()
        t2 = time()

        d_pcd_sr_list.append(d_pcd_sr.copy())
        d_pcd_rs_list.append(d_pcd_rs.copy())
        d_pcd_total_list.append(d_pcd_total.copy())
        d_particle_sr_list.append(d_particle_sr.copy())
        d_particle_rs_list.append(d_particle_rs.copy())
        d_particle_total_list.append(d_particle_total.copy())

        del mpm_env, env

data_dict_list = [
    {'mean': np.mean(d_pcd_sr_list, axis=0), 'lower': np.min(d_pcd_sr_list, axis=0), 'upper': np.max(d_pcd_sr_list, axis=0)},
    {'mean': np.mean(d_pcd_rs_list, axis=0), 'lower': np.min(d_pcd_rs_list, axis=0), 'upper': np.max(d_pcd_rs_list, axis=0)},
    {'mean': np.mean(d_pcd_total_list, axis=0), 'lower': np.min(d_pcd_total_list, axis=0), 'upper': np.max(d_pcd_total_list, axis=0)},
    {'mean': np.mean(d_particle_sr_list, axis=0), 'lower': np.min(d_particle_sr_list, axis=0), 'upper': np.max(d_particle_sr_list, axis=0)},
    {'mean': np.mean(d_particle_rs_list, axis=0), 'lower': np.min(d_particle_rs_list, axis=0), 'upper': np.max(d_particle_rs_list, axis=0)},
    {'mean': np.mean(d_particle_total_list, axis=0), 'lower': np.min(d_particle_total_list, axis=0), 'upper': np.max(d_particle_total_list, axis=0)},
]

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

N = len(data_dict_list[0]["mean"])
x = [i for i in range(N)]
for i in range(len(data_dict_list)):
    case_data = data_dict_list[i]
    # plt.fill_between(x, case_data["upper"], case_data["lower"], alpha=0.3, color=colors[i], label='_nolegend_')
    plt.plot(x, case_data["mean"], color=colors[i])

plt.title('Different loss over manipulation trajectory')
plt.ylabel('Avg. Losses')
plt.xlabel('Time Step')
plt.legend(['d_pcd_sr', 'd_pcd_rs', 'd_pcd_total', 'd_particle_sr', 'd_particle_rs', 'd_particle_total'], loc='upper right')
plt.savefig(os.path.join(script_path, '..', 'loss-comparison', 'exponential-chamfer-tr-1.pdf'), bbox_inches='tight', dpi=500)
