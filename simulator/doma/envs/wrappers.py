import gym
import taichi as ti
import numpy as np
import open3d as o3d
from gym.spaces import Discrete, Box
from doma.engine.configs.macros import DTYPE_NP, DTYPE_TI, SAND
from doma.engine.loss_function.emd_loss_external import compute_emd_loss_external
from doma.envs.planting_env import make_env
from doma.engine.utils.misc import set_parameters, get_gpu_memory


class SingleSkillEnv(gym.Env):
    def __init__(self, ti_env_cfg, gym_env_config, seed=None, logger=None):
        self.logger = logger
        self.mpm_env_cfg = ti_env_cfg['env_cfg']
        self.loss_cfg = ti_env_cfg['loss_cfg']
        self.cam_cfg = ti_env_cfg['cam_cfg']
        self.ti_cfg = ti_env_cfg['ti_cfg']
        self.mpm_env, self.mpm_env_init_state = None, None
        self.mpm_env, self.mpm_env_init_state = self.recreate_mpm_env()

        if seed is not None:
            self.seed(seed)

        self.agent_init_state = self.mpm_env_init_state['agent']

        self.step_count = 0
        self.render_skill = gym_env_config['render_skill']
        self.obs_mode = gym_env_config['obs_mode']
        self.reward_scale = gym_env_config['reward_scale']
        self.horizon = gym_env_config['horizon']
        self.action_space = Box(low=gym_env_config['action_min'],
                                high=gym_env_config['action_max'],
                                shape=(gym_env_config['action_dim'],),
                                dtype=DTYPE_NP)
        self.skill_generation_func = gym_env_config['skill_generation_func']

        self.distance_threshold = 0.01
        self.goal_conditioned_reward_function = compute_emd_loss_external

    def recreate_mpm_env(self):
        if self.mpm_env is not None:
            self.mpm_env.simulator.clear_ckpt()

        ti.reset()
        ti.init(arch=self.ti_cfg['arch'], device_memory_GB=self.ti_cfg['device_memory_GB'],
                default_fp=DTYPE_TI, fast_math=self.ti_cfg['fast_math'],
                random_seed=self.ti_cfg['random_seed'])
        env, mpm_env, init_state = make_env(self.mpm_env_cfg,
                                            self.loss_cfg,
                                            cam_cfg=self.cam_cfg,
                                            debug_grad=False,
                                            logger=self.logger)
        set_parameters(mpm_env, SAND,
                       e=self.mpm_env_cfg['best_params']['E'],
                       nu=self.mpm_env_cfg['best_params']['nu'],
                       rho=self.mpm_env_cfg['best_params']['rho'],
                       sand_friction_angle=self.mpm_env_cfg['best_params']['sand_angle'])
        return mpm_env, init_state['state']

    def step(self, action):
        skill_trajectory = self.skill_generation_func(action,
                                                      dt=self.mpm_env.simulator.dt_global)
        for i in range(len(skill_trajectory)):
            self.mpm_env.step(skill_trajectory[i])
            if self.render_skill:
                self.render(mode='human')
        self.mpm_env.simulator.agent.set_state(self.mpm_env.simulator.cur_substep_local,
                                               self.agent_init_state)
        if self.render_skill:
            self.render(mode='human')
        obs = self.render(mode=self.obs_mode)
        agent_state = self.mpm_env.simulator.agent.get_state(self.mpm_env.simulator.cur_substep_local)
        loss_info = self.mpm_env.get_final_loss()
        reward = loss_info['total_loss'] * self.reward_scale
        self.step_count += 1
        done = self.step_count >= self.horizon
        return {
            'observation': obs.copy(),
            'agent_state': agent_state.copy(),
            'desired_goal': self.up_sample_pcd(self.mpm_env.loss.target_pcd_points_np.copy()),
            'achieved_goal': obs.copy()
        }, reward, done, loss_info

    def reset(self):
        self.mpm_env, self.mpm_env_init_state = self.recreate_mpm_env()

        self.mpm_env.set_state(self.mpm_env_init_state, grad_enabled=False)
        self.step_count = 0
        obs = self.render(mode=self.obs_mode)
        agent_state = self.mpm_env.simulator.agent.get_state(self.mpm_env.simulator.cur_substep_local)
        return {
            'observation': obs.copy(),
            'agent_state': agent_state.copy(),
            'desired_goal': self.up_sample_pcd(self.mpm_env.loss.target_pcd_points_np.copy()),
            'achieved_goal': obs.copy()
        }

    def render(self, mode='human'):
        return self.mpm_env.render(mode=mode)

    def up_sample_pcd(self, pcd_np):
        target_size = self.mpm_env.loss.height_grid_res ** 2
        current_size = pcd_np.shape[0]
        if current_size >= target_size:
            return pcd_np
        random_indices = np.random.choice(current_size, size=(target_size - current_size), replace=False)
        new_points = []
        for i in range(len(random_indices)):
            new_points.append(pcd_np[random_indices[i]] + np.random.normal(0, 0.0005, size=(3,)))
        new_points = np.asarray(new_points, dtype=DTYPE_NP)
        return np.concatenate((pcd_np, new_points), axis=0)


class HybridActionEnv(SingleSkillEnv):
    def __init__(self, mem_env, gym_env_config, seed=None, logger=None):
        SingleSkillEnv.__init__(mem_env, gym_env_config, seed, logger)
        self.discrete_action_space = Discrete(n=gym_env_config['n_discrete_action'])
        self.continuous_action_space = Box(low=gym_env_config['continuous_action_min'],
                                           high=gym_env_config['continuous_action_max'],
                                           shape=(gym_env_config['dim_continuous_action'],),
                                           dtype=DTYPE_NP)

    def step(self, action):
        discrete_action = int(action[0])
        continuous_action = np.asarray(action[1:])
        skill_trajectory = self.skill_generation_func(discrete_action,
                                                      continuous_action,
                                                      dt=self.mpm_env.simulator.dt_global)
        for i in range(len(skill_trajectory)):
            self.mpm_env.step(skill_trajectory[i])
            if self.render_skill:
                self.render(mode='human')

        obs = self.render(mode=self.obs_mode)
        agent_state = self.mpm_env.simulator.agent.get_state(self.mpm_env.simulator.cur_substep_local)
        loss_info = self.mpm_env.get_final_loss()
        reward = loss_info['total_loss'] * self.reward_scale
        self.step_count += 1
        done = self.step_count >= self.horizon
        return {
            'observation': obs.copy(),
            'agent_state': agent_state.copy(),
            'desired_goal': self.up_sample_pcd(self.mpm_env.loss.target_pcd_points_np.copy()),
            'achieved_goal': obs.copy()
        }, reward, done, loss_info


class TrajectoryEnv(SingleSkillEnv):
    def __init__(self, mem_env, gym_env_config, seed=None, logger=None):
        SingleSkillEnv.__init__(mem_env, gym_env_config, seed, logger)

    def step(self, action):
        action *= 0.004

        self.mpm_env.step(action)

        obs = self.render(mode=self.obs_mode)
        agent_state = self.mpm_env.simulator.agent.get_state(self.mpm_env.simulator.cur_substep_local)
        loss_info = self.mpm_env.get_final_loss()
        reward = loss_info['total_loss'] * self.reward_scale
        self.step_count += 1
        done = self.step_count >= self.horizon
        return {
            'observation': obs.copy(),
            'agent_state': agent_state.copy(),
            'desired_goal': self.up_sample_pcd(self.mpm_env.loss.target_pcd_points_np.copy()),
            'achieved_goal': obs.copy()
        }, reward, done, loss_info


class FakeEnv(gym.Env):
    def __init__(self, gym_env_config, onestep=False, seed=None, logger=None):
        if seed is not None:
            self.seed(seed)
        self.pcd_file_path = gym_env_config['pcd_file_path']
        self.pcd_np = np.asarray(o3d.io.read_point_cloud(self.pcd_file_path).voxel_down_sample(voxel_size=0.005).points,
                                 dtype=DTYPE_NP)

        self.step_count = 0
        self.obs_mode = gym_env_config['obs_mode']
        self.reward_scale = gym_env_config['reward_scale']
        if onestep:
            self.horizon = 1
            self.action_space = Box(low=gym_env_config['action_min'],
                                    high=gym_env_config['action_max'],
                                    shape=(gym_env_config['action_dim'],),
                                    dtype=DTYPE_NP)
        else:
            self.horizon = gym_env_config['horizon']
            self.discrete_action_space = Discrete(n=gym_env_config['n_discrete_action'])
            self.continuous_action_space = Box(low=gym_env_config['continuous_action_min'],
                                               high=gym_env_config['continuous_action_max'],
                                               shape=(gym_env_config['dim_continuous_action'],),
                                               dtype=DTYPE_NP)
        self.skill_generation_func = gym_env_config['skill_generation_func']

        self.logger = logger
        self.distance_threshold = 0.01
        self.goal_conditioned_reward_function = compute_emd_loss_external

    def seed(self, seed=None):
        super(FakeEnv, self).seed(seed)

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= self.horizon
        reward = self.goal_conditioned_reward_function(self.pcd_np, self.pcd_np, self.distance_threshold)
        return {
            'observation': self.pcd_np.copy(),
            'agent_state': np.ones(shape=(6,), dtype=np.float32),
            'desired_goal': self.pcd_np.copy(),
            'achieved_goal': self.pcd_np.copy()
        }, reward, done, {}

    def reset(self):
        self.step_count = 0
        return {
            'observation': self.pcd_np.copy(),
            'agent_state': np.ones(shape=(6,), dtype=np.float32),
            'desired_goal': self.pcd_np.copy(),
            'achieved_goal': self.pcd_np.copy()
        }

    def render(self, mode='human'):
        pass
