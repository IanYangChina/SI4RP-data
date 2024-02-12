import os
import numpy as np
from doma.engine.base_envs.gnn_env import make_env
from vedo import Points, show, Mesh
# from doma.engine.utils.misc import get_gpu_memory
# import psutil
# process = psutil.Process(os.getpid())

script_path = os.path.dirname(os.path.realpath(__file__))
motion_ind = 1
agent = 'Rectangle'
agent_init_euler = (0, 0, 45) if agent == 'Rectangle' else (0, 0, 0)
data_ind = 0
training_data_path = os.path.join(script_path, '..', f'data-motion-{motion_ind}', f'eef-{agent.lower()}')

data_cfg = {
    'data_path': training_data_path,
    'data_ind': str(data_ind),
    'trajectory_path': os.path.join(script_path, '..', 'trajectories', f'tr_{motion_ind}_v_dt_0.01.npy'),
    'trajectory_dt': 0.01,
}
env_cfg = {
    'p_density': 2e7,
    'agent_name': agent,
    'agent_init_euler': agent_init_euler,
}

env = make_env(data_cfg, env_cfg)

x_init = env.particles['init_obj_mesh_particles']
RGBA = np.zeros((len(x_init), 4))
RGBA[:, 0] = x_init[:, 2] / x_init[:, 2].max() * 255
RGBA[:, 1] = x_init[:, 2] / x_init[:, 2].max() * 255
RGBA[:, -1] = 255
x_init = Points(x_init, r=12, c=RGBA)

x_end = env.particles['target_obj_mesh_particles']
x_end[:, 1] += 0.1
RGBA = np.zeros((len(x_end), 4))
RGBA[:, -1] = 255
RGBA[:, 0] = x_end[:, 2] / x_end[:, 2].max() * 255
RGBA[:, 1] = x_end[:, 2] / x_end[:, 2].max() * 255
x_end = Points(x_end, r=12, c=RGBA)

eef_init = env.particles['agent_init_particles']
RGBA = np.zeros((len(eef_init), 4))
RGBA[:, -1] = 255
RGBA[:, 0] = 225
eef_init = Points(eef_init, r=12, c=RGBA)

eef_end = env.particles['agent_end_particles']
eef_end[:, 1] += 0.1
RGBA = np.zeros((len(eef_end), 4))
RGBA[:, -1] = 255
RGBA[:, 1] = 255
eef_end = Points(eef_end, r=12, c=RGBA)

show([x_init, x_end, eef_init, eef_end], __doc__, axes=True).close()