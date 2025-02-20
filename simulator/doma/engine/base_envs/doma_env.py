import os
import gym
import numpy as np
from gym.spaces import Box
from doma.engine.configs.macros import DTYPE_NP
from doma.engine.base_envs.mpm_env import MPMEnv
import doma.engine.utils.misc as misc_utils


class DomaEnv(gym.Env):
    """
    Base env class.
    """
    def __init__(self, horizon=500, horizon_action=500, dt_global=0.001, n_substeps=10, grid_scale=1,
                 n_obs_ptcls_per_body=200, collide_type='rigid', grad_op='clip',
                 has_loss=True, loss_type='diff', seed=None,
                 action_min=-5.0, action_max=5.0, problem_dim=3, ptcl_density=1e6,
                 debug_grad=False, logger=None):
        if seed is not None:
            self.seed(seed)

        self.horizon = horizon
        self.horizon_action = horizon_action
        self.n_obs_ptcls_per_body = n_obs_ptcls_per_body
        self.has_loss = has_loss
        self.loss_type = loss_type
        self.action_range = np.array([action_min, action_max])

        self.logger = logger

        # create a taichi env
        self.mpm_env = MPMEnv(
            dim=problem_dim,
            grid_scale=grid_scale,
            particle_density=ptcl_density,
            horizon=self.horizon,
            dt_global=dt_global,
            n_substeps=n_substeps,
            collide_type=collide_type,
            grad_op=grad_op,
            debug_grad=debug_grad,
            logger=self.logger
        )
        self.build_env()
        self.gym_misc()

    def seed(self, seed=None):
        super(DomaEnv, self).seed(seed)
        misc_utils.set_random_seed(seed)

    def build_env(self):
        self.setup_boundary()
        self.setup_statics()
        self.setup_bodies()
        self.setup_agent()
        if not misc_utils.is_on_server():
            self.setup_renderer()
        if self.has_loss:
            self.setup_loss()

        self.mpm_env.build()
        self._init_state = self.mpm_env.get_state()

        # print(f'===>  {type(self).__name__} built successfully.')

    def setup_agent(self):
        pass

    def setup_statics(self):
        # add static mesh-based objects in the scene
        pass

    def setup_bodies(self):
        # add fluid/object bodies
        pass

    def setup_boundary(self):
        pass

    def setup_renderer(self):
        pass

    def setup_loss(self):
        pass

    def gym_misc(self):
        if self.loss_type == 'default':
            self.horizon = self.horizon_action
        obs = self.reset()
        self.observation_space = Box(DTYPE_NP(-np.inf), DTYPE_NP(np.inf), obs.shape, dtype=DTYPE_NP)
        if self.mpm_env.agent is not None:
            self.action_space = Box(DTYPE_NP(self.action_range[0]), DTYPE_NP(self.action_range[1]),
                                    (self.mpm_env.agent.action_dim,), dtype=DTYPE_NP)
        else:
            self.action_space = None

    def reset(self):
        self.mpm_env.set_state(**self._init_state)
        return self._get_obs()

    def _get_obs(self):
        state = self.mpm_env.get_state_RL()
        obs = []

        if 'x' in state:
            for body_id in range(self.mpm_env.particles['bodies']['n']):
                body_n_particles = self.mpm_env.particles['bodies']['n_particles'][body_id]
                body_particle_ids = self.mpm_env.particles['bodies']['particle_ids'][body_id]

                step_size = max(1, body_n_particles // self.n_obs_ptcls_per_body)
                body_x = state['x'][body_particle_ids][::step_size]
                body_v = state['v'][body_particle_ids][::step_size]

                obs.append(body_x.flatten())
                obs.append(body_v.flatten())

        if 'agent' in state:
            obs += state['agent']

        obs = np.concatenate(obs)
        return obs

    def _get_reward(self):
        loss_info = self.mpm_env.get_step_loss()
        return loss_info['reward']

    def step(self, action):
        action = action.clip(self.action_range[0], self.action_range[1])

        self.mpm_env.step(action)

        obs = self._get_obs()
        reward = self._get_reward()

        assert self.t <= self.horizon
        if self.t == self.horizon:
            done = True
        else:
            done = False

        if np.isnan(reward):
            reward = -1000
            done = True

        info = dict()
        return obs, reward, done, info

    def render(self, mode='human'):
        assert mode in ['human', 'rgb_array', 'point_cloud', 'depth_array']
        return self.mpm_env.render(mode)

    @property
    def t(self):
        return self.mpm_env.t
