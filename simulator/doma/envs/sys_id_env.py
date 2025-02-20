import os
from doma.engine.base_envs.doma_env import DomaEnv
from yacs.config import CfgNode
from doma.engine.configs.macros import *
from doma.engine.loss_function.DPSI_loss import DPSILosses
from doma.engine.loss_function.DPSI_planning_loss import DPSIPlanningLosses
from doma.engine.configs import agent_cfg_dir


class SysIDEnv(DomaEnv):
    def __init__(self, horizon=500, horizon_action=500, dt_global=0.001, n_substeps=10, n_obs_ptcls_per_body=200,
                 has_loss=True, seed=None,
                 agent_cfg_file='rectangle_eef.yaml', agent_init_pos=(0.25, 0.25, 0.015), agent_init_euler=(0.0, 0.0, 0.0),
                 render_agent=True, camera_cfg=None,
                 mesh_file=None, material_id=0, initial_pos=(0.25, 0.25, 0.015), voxelise_res=256,
                 loss_cfg=None,
                 action_min=-1.0, action_max=1.0, problem_dim=3, ptcl_density=1e6,
                 debug_grad=False, logger=None):
        self.agent_cfg_file_path = os.path.join(agent_cfg_dir, agent_cfg_file)
        self.mesh_file = mesh_file
        self.material_id = material_id
        assert self.material_id in [0, 1, 2, 3]
        self.initial_pos = initial_pos
        self.voxelise_res = voxelise_res
        self.loss_cfg = loss_cfg
        assert self.loss_cfg is not None
        self.agent_init_pos = agent_init_pos
        self.agent_init_euler = agent_init_euler
        self.render_agent = render_agent
        self.camera_cfg = camera_cfg
        self.logger = logger
        super(SysIDEnv, self).__init__(horizon=horizon, horizon_action=horizon_action,
                                       dt_global=dt_global, n_substeps=n_substeps,
                                       n_obs_ptcls_per_body=n_obs_ptcls_per_body,
                                       has_loss=has_loss, seed=seed, grad_op='none',
                                       action_min=action_min, action_max=action_max,
                                       problem_dim=problem_dim, ptcl_density=ptcl_density,
                                       debug_grad=debug_grad, logger=self.logger)

    def setup_agent(self):
        agent_cfg = CfgNode(new_allowed=True)
        agent_cfg.merge_from_file(self.agent_cfg_file_path)
        # manipulator init pos
        agent_cfg.effectors[0]['params']['init_pos'] = self.agent_init_pos
        agent_cfg.effectors[0]['params']['init_euler'] = self.agent_init_euler
        self.mpm_env.setup_agent(agent_cfg)
        self.agent = self.mpm_env.agent
        if self.logger is not None:
            self.agent.logger = self.logger

    def setup_statics(self):
        self.mpm_env.add_static(
            file='table_surface.obj',
            pos=(0.0, 0.0, 0.01),
            euler=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            material=TABLE,
            has_dynamics=False,
        )

    def setup_bodies(self):
        if self.mesh_file is None:
            raise ValueError('mesh_file is not specified')
        self.mpm_env.add_body(
            type='mesh',
            filling='grid',
            save_voxelised=True,
            voxelize_res=self.voxelise_res,
            file=self.mesh_file,
            pos=self.initial_pos,
            material=self.material_id
        )

    def setup_boundary(self):
        self.mpm_env.setup_boundary(
            type='cube',
            lower=(0.0, 0.0, 0.01),
            upper=(0.5, 0.5, 0.5)
        )

    def setup_renderer(self):
        gl_render = False
        # gl_render = True
        if gl_render:
            self.mpm_env.setup_renderer(
                type='GL',
                # render_particle=True,
                camera_pos=(-0.15, 2.82, 2.5),
                camera_lookat=(0.5, 0.5, 0.5),
                fov=30,
                light_pos=(0.5, 5.0, 0.55),
                light_lookat=(0.5, 0.5, 0.49),
            )
        else:
            if self.camera_cfg is not None:
                cam_pos = self.camera_cfg['pos']
                cam_lookat = self.camera_cfg['lookat']
                fov = self.camera_cfg['fov']
                lights = self.camera_cfg['lights']
                particle_radius = self.camera_cfg['particle_radius']
                res = self.camera_cfg['res']
                cam_euler = self.camera_cfg['euler']
                cam_focal_length = self.camera_cfg['focal_length']
            else:
                cam_pos = (0.3, -0.1, 0.1)
                cam_lookat = (0.25, 0.25, 0.05)
                fov = 30
                lights = [{'pos': (0.5, -1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
                          {'pos': (0.5, -1.5, 1.5), 'color': (0.5, 0.5, 0.5)}]
                particle_radius = 0.003
                res = (640, 480)
                cam_euler = (135, 0, 180)
                cam_focal_length = 0.01

            self.mpm_env.setup_renderer(
                type='GGUI',
                res=res,
                pcd_gen_res=60,
                # render_particle=True,
                camera_pos=cam_pos,
                camera_lookat=cam_lookat,
                camera_fov=fov,
                camera_euler=cam_euler,
                camera_focal_length=cam_focal_length,
                lights=lights,
                render_agent=self.render_agent,
                render_world_frame=False,
                particle_radius=particle_radius
            )

    def setup_loss(self):
        if self.loss_cfg['planning']:
            self.mpm_env.setup_loss(
                loss_cls=DPSIPlanningLosses,
                matching_mat=self.material_id,
                target_pcd_height_map_path=self.loss_cfg['target_pcd_height_map_path'],
                height_map_res=self.loss_cfg['height_map_res'],
                height_map_size=self.loss_cfg['height_map_size'],
                logger=self.logger
            )
        else:
            self.mpm_env.setup_loss(
                loss_cls=DPSILosses,
                matching_mat=self.material_id,
                exponential_distance=self.loss_cfg['exponential_distance'],
                averaging_loss=self.loss_cfg['averaging_loss'],
                point_distance_rs_loss=self.loss_cfg['point_distance_rs_loss'],
                point_distance_sr_loss=self.loss_cfg['point_distance_sr_loss'],
                target_pcd_path=self.loss_cfg['target_pcd_path'],
                target_pcd_offset=self.loss_cfg['target_pcd_offset'],
                down_sample_voxel_size=self.loss_cfg['down_sample_voxel_size'],
                particle_distance_rs_loss=self.loss_cfg['particle_distance_rs_loss'],
                particle_distance_sr_loss=self.loss_cfg['particle_distance_sr_loss'],
                target_mesh_path=self.loss_cfg['target_mesh_file'],
                voxelize_res=self.loss_cfg['voxelise_res'],
                target_mesh_start_pos=self.loss_cfg['target_mesh_start_pos'],
                target_mesh_offset=self.loss_cfg['target_mesh_offset'],
                particle_density=self.loss_cfg['ptcl_density'],
                load_height_map=self.loss_cfg['load_height_map'],
                target_pcd_height_map_path=self.loss_cfg['target_pcd_height_map_path'],
                height_map_loss=self.loss_cfg['height_map_loss'],
                height_map_res=self.loss_cfg['height_map_res'],
                height_map_size=self.loss_cfg['height_map_size'],
                emd_point_distance_loss=self.loss_cfg['emd_point_distance_loss'],
                emd_particle_distance_loss=self.loss_cfg['emd_particle_distance_loss'],
                logger=self.logger
            )


def make_env(data_cfg, env_cfg, loss_config, cam_cfg=None, debug_grad=False, logger=None):
    # Loading data
    data_path = data_cfg['data_path']
    data_ind = data_cfg['data_ind']
    obj_start_mesh_file_path = os.path.join(data_path, f'mesh_{data_ind}0_repaired_normalised.obj')
    if not os.path.exists(obj_start_mesh_file_path):
        raise ValueError(f"File not found: {obj_start_mesh_file_path}")
    obj_start_centre_top_normalised = np.load(
        os.path.join(data_path, f'mesh_{data_ind}0_repaired_normalised_centre_top.npy')).astype(DTYPE_NP)
    obj_start_initial_pos = np.array([0.25, 0.25, obj_start_centre_top_normalised[-1] + 0.01], dtype=DTYPE_NP)
    try:
        agent_init_pos_xy_offset = env_cfg['agent_init_pos_xy_offset']
    except KeyError:
        agent_init_pos_xy_offset = (0.0, 0.0)
    agent_init_pos = (0.25 + agent_init_pos_xy_offset[0],
                      0.25 + agent_init_pos_xy_offset[1],
                      2*obj_start_centre_top_normalised[-1] + 0.01)

    # Loss config
    try:
        planning = loss_config['planning']
    except KeyError:
        planning = False
    height_map_res = loss_config['height_map_res']
    if planning:
        target_ind = loss_config['target_ind']
        try:
            target_hm_path = data_cfg['target_hm_path']
        except KeyError:
            target_hm_path = os.path.join(data_path, '..', '..', 'data-planning-targets')
        loss_config.update({
            'target_pcd_height_map_path': os.path.join(target_hm_path,
                                                       f'target_pcd_height_map-{target_ind}-res{str(height_map_res)}-vdsize{str(0.001)}.npy'),
        })
    else:
        obj_start_centre_real = np.load(os.path.join(data_path, f'mesh_{data_ind}0_repaired_centre.npy')).astype(DTYPE_NP)
        obj_end_pcd_file_path = os.path.join(data_path, f'pcd_{data_ind}1.ply')
        obj_end_mesh_file_path = os.path.join(data_path, f'mesh_{data_ind}1_repaired_normalised.obj')
        obj_end_centre_real = np.load(os.path.join(data_path, f'mesh_{data_ind}1_repaired_centre.npy'))
        loss_config.update({
            'target_pcd_path': obj_end_pcd_file_path,
            'target_pcd_offset': (-obj_start_centre_real + obj_start_initial_pos),
            'target_mesh_file': obj_end_mesh_file_path,
            'target_mesh_start_pos': obj_start_initial_pos,
            'target_mesh_offset': (obj_end_centre_real - obj_start_centre_real),
            'target_pcd_height_map_path': os.path.join(data_path,
                                                       f'target_pcd_height_map-{data_ind}-res{str(height_map_res)}-vdsize{str(0.001)}.npy'),
        })

    # Environment config
    env = SysIDEnv(ptcl_density=env_cfg['p_density'],
                   horizon=env_cfg['horizon'], dt_global=env_cfg['dt_global'], n_substeps=env_cfg['n_substeps'],
                   material_id=env_cfg['material_id'], voxelise_res=1080,
                   mesh_file=obj_start_mesh_file_path, initial_pos=obj_start_initial_pos,
                   loss_cfg=loss_config,
                   agent_cfg_file=env_cfg['agent_name']+'_eef.yaml', agent_init_pos=agent_init_pos, agent_init_euler=env_cfg['agent_init_euler'],
                   render_agent=True, camera_cfg=cam_cfg,
                   debug_grad=debug_grad, logger=logger)
    env.reset()
    mpm_env = env.mpm_env
    init_state = mpm_env.get_state()

    return env, mpm_env, init_state
