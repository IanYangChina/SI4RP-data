import taichi as ti
import numpy as np
import pickle as pkl
import uuid
import os
import torch
from doma.engine.configs.macros import *
from doma.engine.object.boundaries import create_boundary


@ti.data_oriented
class MPMSimulator:
    def __init__(self, dim, grid_scale, gravity, horizon, dt_global, n_substeps, max_substeps_global, ckpt_dest,
                 collide_type='rigid', grad_op='clip', debug_grad=False, logger=None, tb_logger=None):
        self.debug_grad = debug_grad
        self.logger = logger
        self.tb_logger = tb_logger
        self.log_substep_grad = False
        
        self.dim = dim
        self.ckpt_dest = ckpt_dest
        self.sim_id = str(uuid.uuid4())
        self.gravity = ti.Vector(gravity)

        self.n_grid = int(64 * grid_scale)
        self.dx = 1 / self.n_grid
        self.inv_dx = float(self.n_grid)
        self.dt_global = dt_global
        self.n_substeps = n_substeps
        self.dt = self.dt_global / self.n_substeps
        self.p_vol = 0.0015 ** self.dim
        self.res = (self.n_grid,) * self.dim
        self.max_substeps_local = self.n_substeps * 2
        self.max_substeps_global = max_substeps_global
        self.horizon = horizon
        self.trajectory_length = horizon
        self.max_steps_local = int(self.max_substeps_local / self.n_substeps)

        assert self.n_substeps * self.horizon < self.max_substeps_global
        assert self.max_substeps_local % self.n_substeps == 0

        self.agent = None
        self.has_agent = False
        self.boundary = None
        self.has_particles = False
        self.has_rigid_bodies = False
        self.n_particles = 0
        self.n_particles_per_mat = np.zeros(shape=(NUM_MATERIAL,))

        self.collide_type = collide_type
        assert self.collide_type in ['rigid', 'soft', 'toi']

        self.grad_op = grad_op
        assert self.grad_op in ['clip', 'normalize', 'dynamic-scale', 'none']
        self.grad_min = -1e4
        self.grad_max = 1e4
        self.grad_scale = 10.0

    def setup_boundary(self, **kwargs):
        self.boundary = create_boundary(**kwargs)

    def build(self, agent, statics, particles):
        # default boundary
        if self.boundary is None:
            self.boundary = create_boundary()

        # statics
        self.n_statics = len(statics)
        self.has_statics = False
        if self.n_statics > 0:
            self.has_statics = True
        self.statics = statics

        # particles and bodies
        if particles is not None:
            self.has_particles = True
            self.n_particles = len(particles['x'])
            self.n_particles_per_mat = ti.field(ti.i32, shape=(NUM_MATERIAL,))
            self.n_particles_per_mat.fill(0)
            self.setup_particle_fields()
            self.setup_grid_fields()
            self.setup_ckpt_vars()
            self.init_particle_params_kernel()
            self.init_system_param_kernel()
            self.init_particles_and_bodies(particles)
            if self.n_particles_per_mat[RIGID] > 0:
                self.has_rigid_bodies = True
        else:
            self.has_particles = False
            self.n_particles = 0

        # agent
        self.agent = agent
        if self.agent is not None:
            self.has_agent = True

        # misc
        self.cur_substep_global = 0
        self.disable_grad()  # grad disabled by default

    def setup_particle_fields(self):
        # particle state
        particle_state = ti.types.struct(
            x=ti.types.vector(self.dim, DTYPE_TI),  # position
            v=ti.types.vector(self.dim, DTYPE_TI),  # velocity
            C=ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # affine velocity field
            F=ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # deformation gradient
            F_tmp=ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # temp deformation gradient
            U=ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # SVD
            V=ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # SVD
            S=ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # SVD
            Jp=DTYPE_TI,  # Jp, plastic deformation volume ratio, for sand only
            toi=DTYPE_TI,  # time of impact
            delta_x=ti.types.vector(self.dim, DTYPE_TI),  # delta x for collision
            pre_collide_v=ti.types.vector(self.dim, DTYPE_TI),  # collision velocity
        )

        # single frame particle state for rendering
        particle_state_render = ti.types.struct(
            x=ti.types.vector(self.dim, DTYPE_TI),
            radius=DTYPE_TI,
        )

        # particle info
        particle_info = ti.types.struct(
            mat=ti.i32,
            mat_cls=ti.i32,
            body_id=ti.i32,
        )

        # system parameters
        particle_param = ti.types.struct(
            E=DTYPE_TI,
            nu=DTYPE_TI,
            rho=DTYPE_TI,
            mu_temp=DTYPE_TI,
            mu_temp_2=DTYPE_TI,
            lam_temp=DTYPE_TI,
            lam_temp_2=DTYPE_TI,
        )
        system_param = ti.types.struct(
            manipulator_friction=DTYPE_TI,
            ground_friction=DTYPE_TI,
            container_friction=DTYPE_TI,
            yield_stress=DTYPE_TI,
            sand_friction_angle=DTYPE_TI,
            theta_c=DTYPE_TI,
            theta_s=DTYPE_TI,
        )

        debug_info = ti.types.struct(
            min_epsilon=DTYPE_TI,
            max_epsilon=DTYPE_TI,
            min_delta_gamma=DTYPE_TI,
            max_delta_gamma=DTYPE_TI,
            min_Sigma=DTYPE_TI,
            max_Sigma=DTYPE_TI,
            min_F=DTYPE_TI,
            max_F=DTYPE_TI,
            min_stress=DTYPE_TI,
            max_stress=DTYPE_TI,
            min_dSigma_ddeltagamma=DTYPE_TI,
            max_dSigma_ddeltagamma=DTYPE_TI,
            min_dCentre_dSigma=DTYPE_TI,
            max_dCentre_dSigma=DTYPE_TI,
            min_dStress_dSigma=DTYPE_TI,
            max_dStress_dSigma=DTYPE_TI,
            min_dF_dSigma=DTYPE_TI,
            max_dF_dSigma=DTYPE_TI,
            min_dStress_dF=DTYPE_TI,
            max_dStress_dF=DTYPE_TI,
            min_dStress_dmu=DTYPE_TI,
            max_dStress_dmu=DTYPE_TI,
            min_dStress_dlam=DTYPE_TI,
            max_dStress_dlam=DTYPE_TI,
            min_dSigma_dmu=DTYPE_TI,
            max_dSigma_dmu=DTYPE_TI,
            min_dSigma_dlam=DTYPE_TI,
            max_dSigma_dlam=DTYPE_TI,
            min_dSigma_dSand_alpha=DTYPE_TI,
            max_dSigma_dSand_alpha=DTYPE_TI,
            min_dDelta_gamma_dmu=DTYPE_TI,
            max_dDelta_gamma_dmu=DTYPE_TI,
            min_dDelta_gamma_dlam=DTYPE_TI,
            max_dDelta_gamma_dlam=DTYPE_TI,
            min_dDelta_gamma_dSand_alpha=DTYPE_TI,
            max_dDelta_gamma_dSand_alpha=DTYPE_TI,
            min_particle_grad_x=DTYPE_TI,
            max_particle_grad_x=DTYPE_TI,
            min_particle_grad_v=DTYPE_TI,
            max_particle_grad_v=DTYPE_TI,
            min_particle_grad_C=DTYPE_TI,
            max_particle_grad_C=DTYPE_TI,
            min_particle_grad_F=DTYPE_TI,
            max_particle_grad_F=DTYPE_TI,
            min_particle_grad_F_tmp=DTYPE_TI,
            max_particle_grad_F_tmp=DTYPE_TI,
            min_particle_grad_U=DTYPE_TI,
            max_particle_grad_U=DTYPE_TI,
            min_particle_grad_V=DTYPE_TI,
            max_particle_grad_V=DTYPE_TI,
            min_particle_grad_S=DTYPE_TI,
            max_particle_grad_S=DTYPE_TI,
            min_grid_grad_v_in=DTYPE_TI,
            max_grid_grad_v_in=DTYPE_TI,
            min_grid_grad_mass=DTYPE_TI,
            max_grid_grad_mass=DTYPE_TI,
            min_grid_grad_v_out=DTYPE_TI,
            max_grid_grad_v_out=DTYPE_TI,
        )

        # construct fields
        self.particles = particle_state.field(shape=(self.max_substeps_local + 1, self.n_particles), needs_grad=True, layout=ti.Layout.SOA)
        self.particles_render = particle_state_render.field(shape=(self.n_particles,), needs_grad=False, layout=ti.Layout.SOA)
        self.particles_i = particle_info.field(shape=(self.n_particles,), needs_grad=False, layout=ti.Layout.SOA)
        self.particle_param = particle_param.field(shape=(NUM_MATERIAL,), needs_grad=True, layout=ti.Layout.SOA)
        self.system_param = system_param.field(shape=(), needs_grad=True, layout=ti.Layout.SOA)
        self.debug_info = debug_info.field(shape=(), needs_grad=False, layout=ti.Layout.SOA)
        self.agent_collision_delta_pos_norm = ti.field(DTYPE_TI, shape=(), needs_grad=False)
        self.agent_collision_delta_pos = ti.Vector.field(self.dim, DTYPE_TI, shape=(), needs_grad=False)

    def setup_grid_fields(self):
        grid_cell_state = ti.types.struct(
            v_in=ti.types.vector(self.dim, DTYPE_TI),  # input momentum/velocity
            mass=DTYPE_TI,  # mass
            v_out=ti.types.vector(self.dim, DTYPE_TI),  # output momentum/velocity
        )
        self.grid = grid_cell_state.field(shape=(self.max_substeps_local + 1, *self.res), needs_grad=True,
                                          layout=ti.Layout.SOA)

    def setup_ckpt_vars(self):
        if self.ckpt_dest == 'disk':
            # placeholder np array from checkpointing
            self.x_np = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
            self.v_np = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
            self.C_np = np.zeros((self.n_particles, self.dim, self.dim), dtype=DTYPE_NP)
            self.F_np = np.zeros((self.n_particles, self.dim, self.dim), dtype=DTYPE_NP)
        elif self.ckpt_dest == 'cpu' or 'gpu':
            self.ckpt_ram = dict()
        self.actions_buffer = []
        self.setup_ckpt_dir()

    def setup_ckpt_dir(self):
        self.ckpt_dir = os.path.join('/tmp', 'doma', self.sim_id)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def init_particles_and_bodies(self, particles):
        x = particles['x'].astype(DTYPE_NP)
        mat = particles['mat'].astype(np.int32)
        body_id = particles['body_id'].astype(np.int32)
        mat_cls = np.array([MAT_CLASS[mat_i] for mat_i in mat]).astype(np.int32)

        self.init_particles_kernel(x, mat, mat_cls, body_id)
        self.init_bodies(mat_cls, body_id, particles['bodies'])

    @ti.kernel
    def init_system_param_kernel(self):
        self.system_param[None].manipulator_friction = FRICTION[MANIPULATOR]
        self.system_param[None].ground_friction = FRICTION[GROUND]
        self.system_param[None].container_friction = FRICTION[CONTAINER]
        self.system_param[None].yield_stress = YIELD_STRESS
        self.system_param[None].sand_friction_angle = SAND_FRICTION_ANGLE
        self.system_param[None].theta_c = THETA_C
        self.system_param[None].theta_s = THETA_S

        self.debug_info[None].min_epsilon = 1e8
        self.debug_info[None].max_epsilon = -1e8
        self.debug_info[None].min_delta_gamma = 1e8
        self.debug_info[None].max_delta_gamma = -1e8
        self.debug_info[None].min_Sigma = 1e8
        self.debug_info[None].max_Sigma = -1e8
        self.debug_info[None].min_F = 1e8
        self.debug_info[None].max_F = -1e8
        self.debug_info[None].min_stress = 1e8
        self.debug_info[None].max_stress = -1e8
        self.debug_info[None].min_dCentre_dSigma = 1e8
        self.debug_info[None].max_dCentre_dSigma = -1e8
        self.debug_info[None].min_dStress_dSigma = 1e8
        self.debug_info[None].max_dStress_dSigma = -1e8
        self.debug_info[None].min_dF_dSigma = 1e8
        self.debug_info[None].max_dF_dSigma = -1e8
        self.debug_info[None].min_dStress_dF = 1e8
        self.debug_info[None].max_dStress_dF = -1e8
        self.debug_info[None].min_dStress_dmu = 1e8
        self.debug_info[None].max_dStress_dmu = -1e8
        self.debug_info[None].min_dStress_dlam = 1e8
        self.debug_info[None].max_dStress_dlam = -1e8
        self.debug_info[None].min_dSigma_ddeltagamma = 1e8
        self.debug_info[None].max_dSigma_ddeltagamma = -1e8
        self.debug_info[None].min_dSigma_dmu = 1e8
        self.debug_info[None].max_dSigma_dmu = -1e8
        self.debug_info[None].min_dSigma_dlam = 1e8
        self.debug_info[None].max_dSigma_dlam = -1e8
        self.debug_info[None].min_dSigma_dSand_alpha = 1e8
        self.debug_info[None].max_dSigma_dSand_alpha = -1e8
        self.debug_info[None].min_dDelta_gamma_dmu = 1e8
        self.debug_info[None].max_dDelta_gamma_dmu = -1e8
        self.debug_info[None].min_dDelta_gamma_dlam = 1e8
        self.debug_info[None].max_dDelta_gamma_dlam = -1e8
        self.debug_info[None].min_dDelta_gamma_dSand_alpha = 1e8
        self.debug_info[None].max_dDelta_gamma_dSand_alpha = -1e8

        self.debug_info[None].min_particle_grad_x = 1e8
        self.debug_info[None].max_particle_grad_x = -1e8
        self.debug_info[None].min_particle_grad_v = 1e8
        self.debug_info[None].max_particle_grad_v = -1e8
        self.debug_info[None].min_particle_grad_C = 1e8
        self.debug_info[None].max_particle_grad_C = -1e8
        self.debug_info[None].min_particle_grad_F = 1e8
        self.debug_info[None].max_particle_grad_F = -1e8
        self.debug_info[None].min_particle_grad_U = 1e8
        self.debug_info[None].max_particle_grad_U = -1e8
        self.debug_info[None].min_particle_grad_V = 1e8
        self.debug_info[None].max_particle_grad_V = -1e8
        self.debug_info[None].min_particle_grad_S = 1e8
        self.debug_info[None].max_particle_grad_S = -1e8
        self.debug_info[None].min_grid_grad_v_in = 1e8
        self.debug_info[None].max_grid_grad_v_in = -1e8
        self.debug_info[None].min_grid_grad_mass = 1e8
        self.debug_info[None].max_grid_grad_mass = -1e8
        self.debug_info[None].min_grid_grad_v_out = 1e8
        self.debug_info[None].max_grid_grad_v_out = -1e8

    @ti.kernel
    def init_particle_params_kernel(self):
        for i in ti.static(range(NUM_MATERIAL)):
            self.particle_param[i].E = E[i]
            self.particle_param[i].nu = NU[i]
            self.particle_param[i].rho = RHO[i]
            self.particle_param[i].mu_temp = E[i] / (2 * (1 + NU[i]))
            self.particle_param[i].mu_temp_2 = E[i] / (2 * (1 + NU[i]))
            self.particle_param[i].lam_temp = E[i] * NU[i] / ((1 + NU[i]) * (1 - 2 * NU[i]))
            self.particle_param[i].lam_temp_2 = E[i] * NU[i] / ((1 + NU[i]) * (1 - 2 * NU[i]))

    @ti.kernel
    def init_particles_kernel(
            self,
            x: ti.types.ndarray(),
            mat: ti.types.ndarray(),
            mat_cls: ti.types.ndarray(),
            body_id: ti.types.ndarray()
    ):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                self.particles[0, i].x[j] = x[i, j]
            self.particles[0, i].v = ti.Vector.zero(DTYPE_TI, self.dim)
            self.particles[0, i].F = ti.Matrix.identity(DTYPE_TI, self.dim)
            self.particles[0, i].C = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
            self.particles[0, i].Jp = 0
            self.particles[0, i].toi = 0
            self.particles[0, i].delta_x = ti.Vector.zero(DTYPE_TI, self.dim)
            self.particles[0, i].pre_collide_v = ti.Vector.zero(DTYPE_TI, self.dim)

            self.particles_i[i].mat = mat[i]
            self.particles_i[i].mat_cls = mat_cls[i]
            self.particles_i[i].body_id = body_id[i]
            self.n_particles_per_mat[mat[i]] += 1

    def init_bodies(self, mat_cls, body_id, bodies):
        self.n_bodies = bodies['n']
        assert self.n_bodies == np.max(body_id) + 1

        # body state, for rigidity enforcement
        body_state = ti.types.struct(
            COM_t0=ti.types.vector(self.dim, DTYPE_TI),
            COM_t1=ti.types.vector(self.dim, DTYPE_TI),
            H=ti.types.matrix(self.dim, self.dim, DTYPE_TI),
            R=ti.types.matrix(self.dim, self.dim, DTYPE_TI),
            U=ti.types.matrix(self.dim, self.dim, DTYPE_TI),
            S=ti.types.matrix(self.dim, self.dim, DTYPE_TI),
            V=ti.types.matrix(self.dim, self.dim, DTYPE_TI),
        )
        # body info
        body_info = ti.types.struct(
            n_particles=ti.i32,
            mat_cls=ti.i32,
            p_volume=ti.f32,
            p_radius=ti.f32,
        )
        self.bodies = body_state.field(shape=(self.n_bodies,), needs_grad=True, layout=ti.Layout.SOA)
        self.bodies_i = body_info.field(shape=(self.n_bodies,), needs_grad=False, layout=ti.Layout.SOA)

        for i in range(self.n_bodies):
            self.bodies_i[i].n_particles = np.sum(body_id == i)
            self.bodies_i[i].mat_cls = mat_cls[body_id == i][0]
            self.bodies_i[i].p_volume = bodies['p_volumes'][i]
            self.bodies_i[i].p_radius = bodies['p_radii'][i]

    def reset_grad(self):
        self.particles.grad.fill(0)
        self.particle_param.grad.fill(0)
        self.system_param.grad.fill(0)
        self.grid.grad.fill(0)

    def enable_grad(self):
        '''
        If grad_enable == True, we do checkpointing when gpu memory is not enough for storing the whole episode.
        '''
        self.grad_enabled = True
        self.cur_substep_global = 0

    def disable_grad(self):
        self.grad_enabled = False
        self.cur_substep_global = 0

    # --------------------------------- MPM part -----------------------------------
    @ti.kernel
    def reset_grid_and_grad(self, f: ti.i32):
        for I in ti.grouped(ti.ndrange(*self.res)):
            self.grid[f, I].v_in.fill(0)
            self.grid[f, I].mass = 0
            self.grid[f, I].v_out.fill(0)
            self.grid.grad[f, I].v_in.fill(0)
            self.grid.grad[f, I].mass = 0
            self.grid.grad[f, I].v_out.fill(0)

    def f_global_to_f_local(self, f_global):
        f_local = f_global % self.max_substeps_local
        return f_local

    def f_local_to_s_local(self, f_local):
        f_local = f_local // self.n_substeps
        return f_local

    def f_global_to_s_local(self, f_global):
        # global substeps to local steps
        f_local = self.f_global_to_f_local(f_global)
        s_local = self.f_local_to_s_local(f_local)
        return s_local

    def f_global_to_s_global(self, f_global):
        s_global = f_global // self.n_substeps
        return s_global

    @property
    def cur_substep_local(self):
        return self.f_global_to_f_local(self.cur_substep_global)

    @property
    def cur_step_local(self):
        return self.f_global_to_s_local(self.cur_substep_global)

    @property
    def cur_step_global(self):
        return self.f_global_to_s_global(self.cur_substep_global)

    @ti.kernel
    def compute_F_tmp(self, f: ti.i32):
        for p in range(self.n_particles):
            self.particles[f, p].F_tmp = (ti.Matrix.identity(DTYPE_TI, self.dim) + self.dt * self.particles[
                f, p].C) @ self.particles[f, p].F

    @ti.kernel
    def svd(self, f: ti.i32):
        for p in range(self.n_particles):
            self.particles[f, p].U, self.particles[f, p].S, self.particles[f, p].V = ti.svd(
                self.particles[f, p].F_tmp, DTYPE_TI)

    @ti.kernel
    def svd_grad(self, f: ti.i32):
        for p in range(self.n_particles):
            self.particles.grad[f, p].F_tmp += self.backward_svd(self.particles.grad[f, p].U,
                                                                 self.particles.grad[f, p].S,
                                                                 self.particles.grad[f, p].V,
                                                                 self.particles[f, p].U, self.particles[f, p].S,
                                                                 self.particles[f, p].V)

    @ti.func
    def backward_svd(self, grad_U, grad_S, grad_V, U, S, V):
        # https://github.com/pytorch/pytorch/blob/ab0a04dc9c8b84d4a03412f1c21a6c4a2cefd36c/tools/autograd/templates/Functions.cpp
        vt = V.transpose()
        ut = U.transpose()
        S_term = U @ grad_S @ vt

        s = ti.Vector.zero(DTYPE_TI, self.dim)
        s = ti.Vector([S[0, 0], S[1, 1], S[2, 2]]) ** 2
        F = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
        for i, j in ti.static(ti.ndrange(self.dim, self.dim)):
            if i == j:
                F[i, j] = 0
            else:
                F[i, j] = 1.0 / self.clamp(s[j] - s[i])
        u_term = U @ ((F * (ut @ grad_U - grad_U.transpose() @ U)) @ S) @ vt
        v_term = U @ (S @ ((F * (vt @ grad_V - grad_V.transpose() @ V)) @ vt))
        return u_term + v_term + S_term

    @ti.func
    def clamp(self, a):
        # remember that we don't support if return in taichi
        # stop the gradient ...
        if a >= 0:
            a = ti.max(a, 1e-8)
        else:
            a = ti.min(a, -1e-8)
        return a

    @ti.func
    def stencil_range(self):
        return ti.ndrange(*((3,) * self.dim))

    @ti.func
    def sand_projection(self, f, p, mu, lam):
        # case I, already in yield surface
        sigma_out = ti.max(self.particles[f, p].S, EPS)  # add this to prevent NaN in extreme cases

        epsilon = ti.Vector([ti.log(sigma_out[0, 0]), ti.log(sigma_out[1, 1]), ti.log(sigma_out[2, 2])])
        for i in ti.static(range(self.dim)):
            ti.atomic_max(self.debug_info[None].max_epsilon, epsilon[i])
            ti.atomic_min(self.debug_info[None].min_epsilon, epsilon[i])
        tr = epsilon.sum() + self.particles[f, p].Jp
        epsilon_hat = epsilon - tr / self.dim
        epsilon_hat_norm = ti.sqrt(epsilon_hat.dot(epsilon_hat) + 1e-8)
        dSigma_ddeltagamma = ti.Matrix.one(DTYPE_TI, self.dim, self.dim)
        dSigma_dmu = ti.Matrix.one(DTYPE_TI, self.dim, self.dim)
        dSigma_dlam = ti.Matrix.one(DTYPE_TI, self.dim, self.dim)
        dSigma_dSand_alpha = ti.Matrix.one(DTYPE_TI, self.dim, self.dim)
        if tr >= 0.0:
            # case II
            self.particles[f, p].Jp = tr
            sigma_out = ti.Matrix.identity(DTYPE_TI, self.dim)
        else:
            sin_sand_friction_angle = ti.sin(self.system_param[None].sand_friction_angle / 180 * np.pi)
            sand_alpha = ti.sqrt(2 / 3) * 2 * sin_sand_friction_angle / (3 - sin_sand_friction_angle)
            self.particles[f, p].Jp = 0.0
            delta_gamma = epsilon_hat_norm + (self.dim * lam + 2 * mu) / (2 * mu) * tr * sand_alpha
            ti.atomic_max(self.debug_info[None].max_delta_gamma, delta_gamma)
            ti.atomic_min(self.debug_info[None].min_delta_gamma, delta_gamma)
            dDelta_gamma_dSand_alpha = (self.dim * lam + 2 * mu) / (2 * mu) * tr
            ti.atomic_max(self.debug_info[None].max_dDelta_gamma_dSand_alpha, dDelta_gamma_dSand_alpha)
            ti.atomic_min(self.debug_info[None].min_dDelta_gamma_dSand_alpha, dDelta_gamma_dSand_alpha)
            dDelta_gamma_dmu = -self.dim * lam * tr * sand_alpha / (2 * mu * mu)
            ti.atomic_max(self.debug_info[None].max_dDelta_gamma_dmu, dDelta_gamma_dmu)
            ti.atomic_min(self.debug_info[None].min_dDelta_gamma_dmu, dDelta_gamma_dmu)
            dDelta_gamma_dlam = 3 * tr * sand_alpha / (2 * mu)
            ti.atomic_max(self.debug_info[None].max_dDelta_gamma_dlam, dDelta_gamma_dlam)
            ti.atomic_min(self.debug_info[None].min_dDelta_gamma_dlam, dDelta_gamma_dlam)

            if delta_gamma > 0:  # yields, case III
                exp_epsilon = ti.exp(epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat)
                sigma_out = ti.Matrix([[exp_epsilon[0], 0.0, 0.0],
                                       [0.0, exp_epsilon[1], 0.0],
                                       [0.0, 0.0, exp_epsilon[2]]], dt=DTYPE_TI)
                for i in ti.static(range(self.dim)):
                    # sigma_out[i, i] = ti.exp(epsilon[i] - delta_gamma / epsilon_hat_norm * epsilon_hat[i])
                    dSigma_ddeltagamma[i, i] = ti.exp(epsilon[i] - delta_gamma / epsilon_hat_norm * epsilon_hat[i]) * -epsilon_hat[i] / epsilon_hat_norm
                    ti.atomic_max(self.debug_info[None].max_dSigma_ddeltagamma, dSigma_ddeltagamma[i, i])
                    ti.atomic_min(self.debug_info[None].min_dSigma_ddeltagamma, dSigma_ddeltagamma[i, i])
                    dSigma_dmu[i, i] = dSigma_ddeltagamma[i, i] * dDelta_gamma_dmu
                    dSigma_dlam[i, i] = dSigma_ddeltagamma[i, i] * dDelta_gamma_dlam
                    dSigma_dSand_alpha[i, i] = dSigma_ddeltagamma[i, i] * dDelta_gamma_dSand_alpha
                    ti.atomic_max(self.debug_info[None].max_dSigma_dmu, dSigma_dmu[i, i])
                    ti.atomic_min(self.debug_info[None].min_dSigma_dmu, dSigma_dmu[i, i])
                    ti.atomic_max(self.debug_info[None].max_dSigma_dlam, dSigma_dlam[i, i])
                    ti.atomic_min(self.debug_info[None].min_dSigma_dlam, dSigma_dlam[i, i])
                    ti.atomic_max(self.debug_info[None].max_dSigma_dSand_alpha, dSigma_dSand_alpha[i, i])
                    ti.atomic_min(self.debug_info[None].min_dSigma_dSand_alpha, dSigma_dSand_alpha[i, i])

        return sigma_out, dSigma_dmu, dSigma_dlam

    @ti.kernel
    def p2g(self, f: ti.i32):
        for p in range(self.n_particles):
            base = (self.particles[f, p].x * self.inv_dx - 0.5).cast(int)
            fx = self.particles[f, p].x * self.inv_dx - base.cast(DTYPE_TI)

            mat = self.particles_i[p].mat
            e = self.particle_param[mat].E
            nu = self.particle_param[mat].nu
            # compute lame parameters
            mu = e / (2 * (1 + nu))
            lam = e * nu / ((1 + nu) * (1 - 2 * nu))
            p_vol = self.bodies_i[self.particles_i[p].body_id].p_volume
            mass = self.particle_param[mat].rho * p_vol

            # Quadratic kernels
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            J = self.particles[f, p].S.determinant()

            # update deformation gradient based on material class
            # For rigid and elastic, we use the original deformation gradient
            F_new = self.particles[f, p].F_tmp
            sig = ti.max(self.particles[f, p].S, EPS)  # add this to prevent NaN in extreme cases
            dSigma_dmu = ti.Matrix.identity(DTYPE_TI, self.dim)
            dSigma_dlam = ti.Matrix.identity(DTYPE_TI, self.dim)

            if self.particles_i[p].mat_cls == MAT_LIQUID:
                F_new = ti.Matrix.identity(DTYPE_TI, self.dim)
                F_new[0, 0] = J
                mu = 0.0
                lam = 27770

            elif self.particles_i[p].mat_cls == MAT_PLASTO_ELASTIC:
                # S_new = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
                # for d in ti.static(range(self.dim)):
                #     S_new[d, d] = ti.min(ti.max(self.particles[f, p].S[d, d], 1 - self.system_param[None].theta_c),
                #                          1 + self.system_param[None].theta_s)
                # # Reconstruct elastic deformation gradient after plasticity
                # F_new = self.particles[f, p].U @ S_new @ self.particles[f, p].V.transpose()
                # Compute von Mises F
                epsilon = ti.Vector([ti.log(sig[0, 0]), ti.log(sig[1, 1]), ti.log(sig[2, 2])])
                epsilon_hat = epsilon - (epsilon.sum() / self.dim)
                epsilon_hat_norm = ti.sqrt(epsilon_hat.dot(epsilon_hat) + 1e-8)
                delta_gamma = epsilon_hat_norm - self.system_param[None].yield_stress / (2 * mu)
                dDelta_gamma_dmu = self.system_param[None].yield_stress / (2 * mu * mu)

                if delta_gamma > 0:  # Yields
                    exp_epsilon = ti.exp(epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat)
                    sig = ti.Matrix([[exp_epsilon[0], 0.0, 0.0],
                                     [0.0, exp_epsilon[1], 0.0],
                                     [0.0, 0.0, exp_epsilon[2]]], dt=DTYPE_TI)
                    dSigma_ddeltagamma = ti.Matrix.identity(DTYPE_TI, self.dim)
                    for i in ti.static(range(self.dim)):
                        dSigma_ddeltagamma[i, i] = ti.exp(epsilon[i] - delta_gamma / epsilon_hat_norm * epsilon_hat[i]) * -epsilon_hat[i] / epsilon_hat_norm
                        ti.atomic_max(self.debug_info[None].max_dSigma_ddeltagamma, dSigma_ddeltagamma[i, i])
                        ti.atomic_min(self.debug_info[None].min_dSigma_ddeltagamma, dSigma_ddeltagamma[i, i])
                    dSigma_dmu = dSigma_ddeltagamma * dDelta_gamma_dmu
                    for i in ti.static(range(self.dim)):
                        ti.atomic_max(self.debug_info[None].max_dSigma_dmu, dSigma_dmu[i, i])
                        ti.atomic_min(self.debug_info[None].min_dSigma_dmu, dSigma_dmu[i, i])

                    F_new = self.particles[f, p].U @ sig @ self.particles[f, p].V.transpose()

                for i in ti.static(range(self.dim)):
                    ti.atomic_max(self.debug_info[None].max_Sigma, sig[i, i])
                    ti.atomic_min(self.debug_info[None].min_Sigma, sig[i, i])
                    dF_dSigma_ii = self.particles[f, p].U[i, i] * self.particles[f, p].V[i, i]
                    ti.atomic_max(self.debug_info[None].max_dF_dSigma, dF_dSigma_ii)
                    ti.atomic_min(self.debug_info[None].min_dF_dSigma, dF_dSigma_ii)

            stress = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)

            # compute stress tensor
            if self.particles_i[p].mat_cls == MAT_SAND:
                sig, dSigma_dmu, dSigma_dlam = self.sand_projection(f, p,
                                                                    mu,
                                                                    lam)
                F_new = self.particles[f, p].U @ sig @ self.particles[f, p].V.transpose()

                # St. Venant-Kirchhoff with Hencky strain
                log_sig_sum = 0.0
                inverse_sig_sum = 0.0
                center = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
                for i in ti.static(range(self.dim)):
                    log_sig_sum += ti.log(sig[i, i])
                    inverse_sig_sum += 1 / sig[i, i]
                    center[i, i] = 2.0 * mu * ti.log(sig[i, i]) * (1 / sig[i, i])
                for i in ti.static(range(self.dim)):
                    center[i, i] += lam * log_sig_sum * (1 / sig[i, i])
                stress = self.particles[f, p].U @ center @ self.particles[f, p].V.transpose() @ F_new.transpose()

                dStress_dF = self.particles[f, p].U @ center @ self.particles[f, p].V.transpose()
                dCentre_dmu = ti.Matrix.identity(DTYPE_TI, self.dim)
                dCentre_dlam = ti.Matrix.identity(DTYPE_TI, self.dim)
                dCentre_dSigma = ti.Matrix.identity(DTYPE_TI, self.dim)
                for i in ti.static(range(self.dim)):
                    dCentre_dmu[i, i] = 2 * ti.log(sig[i, i]) * (1 / sig[i, i])
                    dCentre_dlam[i, i] = log_sig_sum * (1 / sig[i, i])
                    dCentre_dSigma[i, i] = (2 * mu * (- ti.log(sig[i, i]) + 1) / (sig[i, i] * sig[i, i]) -
                                            lam * log_sig_sum / (sig[i, i] * sig[i, i])) + lam * inverse_sig_sum / sig[i, i]
                dStress_dmu = self.particles[f, p].U @ dCentre_dmu @ self.particles[f, p].V.transpose()
                dStress_dlam = self.particles[f, p].U @ dCentre_dlam @ self.particles[f, p].V.transpose()
                dStress_dSigma = self.particles[f, p].U @ dCentre_dSigma @ self.particles[f, p].V.transpose() @ F_new.transpose()

                dStress_dmu_ = dStress_dmu + dStress_dSigma @ dSigma_dmu
                dStress_dlam_ = dStress_dlam
                if self.particles_i[p].mat_cls == MAT_SAND:
                    dStress_dlam_ += dStress_dSigma @ dSigma_dlam

                for i in ti.static(range(self.dim)):
                    dF_dSigma_ii = self.particles[f, p].U[i, i] * self.particles[f, p].V[i, i]
                    ti.atomic_max(self.debug_info[None].max_dF_dSigma, dF_dSigma_ii)
                    ti.atomic_min(self.debug_info[None].min_dF_dSigma, dF_dSigma_ii)
                    ti.atomic_max(self.debug_info[None].max_dCentre_dSigma, dCentre_dSigma[i, i])
                    ti.atomic_min(self.debug_info[None].min_dCentre_dSigma, dCentre_dSigma[i, i])
                    ti.atomic_max(self.debug_info[None].max_dStress_dSigma, dStress_dSigma[i, i])
                    ti.atomic_min(self.debug_info[None].min_dStress_dSigma, dStress_dSigma[i, i])
                    ti.atomic_max(self.debug_info[None].max_Sigma, sig[i, i])
                    ti.atomic_min(self.debug_info[None].min_Sigma, sig[i, i])
                    for j in ti.static(range(self.dim)):
                        ti.atomic_max(self.debug_info[None].max_dStress_dF, dStress_dF[i, j])
                        ti.atomic_min(self.debug_info[None].min_dStress_dF, dStress_dF[i, j])
                        ti.atomic_max(self.debug_info[None].max_dStress_dmu, dStress_dmu_[i, j])
                        ti.atomic_min(self.debug_info[None].min_dStress_dmu, dStress_dmu_[i, j])
                        ti.atomic_max(self.debug_info[None].max_dStress_dlam, dStress_dlam_[i, j])
                        ti.atomic_min(self.debug_info[None].min_dStress_dlam, dStress_dlam_[i, j])
            else:
                # Fixed corotated constitutive model
                r = self.particles[f, p].U @ self.particles[f, p].V.transpose()
                stress = 2 * mu * (F_new - r) @ F_new.transpose() + \
                             ti.Matrix.identity(DTYPE_TI, self.dim) * lam * J * (J - 1)
                dStress_dF = 2 * mu * (F_new.transpose() + F_new.transpose() @ F_new - r * F_new)
                dStress_dlam = J * (J - 1)
                ti.atomic_max(self.debug_info[None].max_dStress_dlam, dStress_dlam)
                ti.atomic_min(self.debug_info[None].min_dStress_dlam, dStress_dlam)
                dStress_dSigma = dStress_dF @ self.particles[f, p].U @ self.particles[f, p].V.transpose()
                dStress_dmu = 2 * (F_new - r) @ F_new.transpose() + dStress_dSigma @ dSigma_dmu
                for i in ti.static(range(self.dim)):
                    ti.atomic_max(self.debug_info[None].max_dStress_dSigma, dStress_dSigma[i, i])
                    ti.atomic_min(self.debug_info[None].min_dStress_dSigma, dStress_dSigma[i, i])
                    ti.atomic_max(self.debug_info[None].max_stress, stress[i, i])
                    ti.atomic_min(self.debug_info[None].min_stress, stress[i, i])
                    for j in ti.static(range(self.dim)):
                        ti.atomic_max(self.debug_info[None].max_dStress_dF, dStress_dF[i, j])
                        ti.atomic_min(self.debug_info[None].min_dStress_dF, dStress_dF[i, j])
                        ti.atomic_max(self.debug_info[None].max_dStress_dmu, dStress_dmu[i, j])
                        ti.atomic_min(self.debug_info[None].min_dStress_dmu, dStress_dmu[i, j])

            for i in ti.static(range(self.dim)):
                ti.atomic_max(self.debug_info[None].max_F, F_new[i, i])
                ti.atomic_min(self.debug_info[None].min_F, F_new[i, i])
                ti.atomic_max(self.debug_info[None].max_stress, stress[i, i])
                ti.atomic_min(self.debug_info[None].min_stress, stress[i, i])
            self.particles[f + 1, p].F = F_new
            stress = (-self.dt * p_vol * 4 * self.inv_dx * self.inv_dx) * stress
            affine = stress + mass * self.particles[f, p].C

            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = (offset.cast(DTYPE_TI) - fx) * self.dx
                weight = ti.cast(1.0, DTYPE_TI)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]

                self.grid[f, base + offset].v_in += weight * (mass * self.particles[f, p].v + affine @ dpos)
                self.grid[f, base + offset].mass += weight * mass

    @ti.kernel
    def grid_op(self, f: ti.i32):
        for I in ti.grouped(ti.ndrange(*self.res)):
            if self.grid[f, I].mass > EPS:
                v_out = (1 / self.grid[f, I].mass) * self.grid[f, I].v_in  # Momentum to velocity
                v_out += self.dt * self.gravity  # gravity

                # collide with agent
                mf = self.system_param[None].manipulator_friction
                if ti.static(self.has_agent):
                    if ti.static(self.agent.collide_type in ['grid', 'both']):
                        v_out = self.agent.collide(f, I * self.dx, v_out, self.dt, mf)

                # collide with statics
                sf = self.system_param[None].container_friction
                if ti.static(self.has_statics):
                    for i in ti.static(range(self.n_statics)):
                        v_out = self.statics[i].collide(I * self.dx, v_out, sf)

                # impose boundary
                gf = self.system_param[None].ground_friction
                _, self.grid[f, I].v_out = self.boundary.impose_x_v(I * self.dx, v_out, gf)

    @ti.kernel
    def g2p(self, f: ti.i32):
        for p in range(self.n_particles):
            base = (self.particles[f, p].x * self.inv_dx - 0.5).cast(int)
            fx = self.particles[f, p].x * self.inv_dx - base.cast(DTYPE_TI)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(DTYPE_TI, self.dim)
            new_C = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = offset.cast(DTYPE_TI) - fx
                g_v = self.grid[f, base + offset].v_out
                weight = ti.cast(1.0, DTYPE_TI)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                new_v += weight * g_v
                new_C += 4 * self.inv_dx * (weight * g_v).outer_product(dpos)

            new_x_tmp = self.particles[f, p].x + self.dt * new_v
            toi = 0.0

            # advect to next frame
            self.particles[f + 1, p].pre_collide_v = new_v
            self.particles[f + 1, p].C = new_C

            collide_v = new_v
            # collide with agent
            mf = self.system_param[None].manipulator_friction
            if ti.static(self.has_agent):
                if ti.static(self.agent.collide_type in ['particle', 'both']):
                    if ti.static(self.collide_type == 'rigid'):
                        collide_v = self.agent.collide(f, new_x_tmp, new_v, self.dt, mf)
                    elif ti.static(self.collide_type == 'soft'):
                        collide_v = self.agent.collide_soft(f, new_x_tmp, new_v, self.dt, mf)
                    else:
                        delta_x, mat_v_toi, toi = self.agent.collide_toi(f, new_x_tmp, new_v, self.dt, mf)
                        collide_v = mat_v_toi
                        self.particles[f, p].delta_x = delta_x

            # impose boundary
            gf = self.system_param[None].ground_friction
            _, collide_v = self.boundary.impose_x_v(new_x_tmp, collide_v, gf)

            # advect to next frame
            self.particles[f, p].toi = toi
            self.particles[f + 1, p].v = collide_v

    def advect(self, f):
        if self.has_rigid_bodies:
            self.reset_rigid_bodies_and_grad()
            self.compute_COM(f)
            self.compute_H(f)
            self.compute_H_svd(f)
            self.compute_R(f)
            self.advect_kernel_with_rigid(f)
        else:
            if self.collide_type != 'toi':
                self.advect_kernel(f)
            else:
                self.advect_kernel_toi(f)

    def advect_grad(self, f):
        if self.has_rigid_bodies:
            self.reset_rigid_bodies_and_grad()
            self.compute_COM(f)
            self.compute_H(f)
            self.compute_H_svd(f)
            self.compute_R(f)
            self.advect_kernel_with_rigid.grad(f)
            if self.debug_grad:
                print(f'******Grads after advect_kernel_with_rigid.grad() at substep {f}, global step {self.cur_step_global}')
                self.print_grads()
            self.compute_R.grad(f)
            if self.debug_grad:
                print(f'******Grads after compute_R.grad() at substep {f}, global step {self.cur_step_global}')
                self.print_grads()
            self.compute_H_svd_grad(f)
            if self.debug_grad:
                print(f'******Grads after compute_H_svd_grad() at substep {f}, global step {self.cur_step_global}')
                self.print_grads()
            self.compute_H.grad(f)  # This causes compilation error in cuda backend, avoid using for the time being.
            if self.debug_grad:
                print(f'******Grads after compute_H.grad() at substep {f}, global step {self.cur_step_global}')
                self.print_grads()
            self.compute_COM.grad(f)
            if self.debug_grad:
                print(f'******Grads after compute_COM.grad() at substep {f}, global step {self.cur_step_global}')
                self.print_grads()
        else:
            if self.collide_type != 'toi':
                self.advect_kernel.grad(f)
            else:
                self.advect_kernel_toi.grad(f)
            if self.debug_grad:
                print(f'******Grads after advect_kernel.grad() at substep {f}, global step {self.cur_step_global}')
                self.print_grads()

    @ti.kernel
    def reset_rigid_bodies_and_grad(self):
        for body_id in range(self.n_bodies):
            if self.bodies_i[body_id].mat_cls == MAT_RIGID:
                self.bodies[body_id].fill(0)
                self.bodies.grad[body_id].fill(0)

    @ti.kernel
    def compute_COM(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particles_i[p].mat_cls == MAT_RIGID:
                body_id = self.particles_i[p].body_id
                self.bodies[body_id].COM_t0 += self.particles[f, p].x / ti.cast(self.bodies_i[body_id].n_particles, DTYPE_TI)
                self.bodies[body_id].COM_t1 += (self.particles[f, p].x + self.dt * self.particles[f + 1, p].v) / ti.cast(self.bodies_i[body_id].n_particles, DTYPE_TI)

    @ti.kernel
    def compute_H(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particles_i[p].mat_cls == MAT_RIGID:
                body_id = self.particles_i[p].body_id
                self.bodies[body_id].H[0, 0] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[0] * (self.particles[f, p].x + self.dt * self.particles[f + 1, p].v - self.bodies[body_id].COM_t1)[0]
                self.bodies[body_id].H[0, 1] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[0] * (self.particles[f, p].x + self.dt * self.particles[f + 1, p].v - self.bodies[body_id].COM_t1)[1]
                self.bodies[body_id].H[0, 2] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[0] * (self.particles[f, p].x + self.dt * self.particles[f + 1, p].v - self.bodies[body_id].COM_t1)[2]
                self.bodies[body_id].H[1, 0] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[1] * (self.particles[f, p].x + self.dt * self.particles[f + 1, p].v - self.bodies[body_id].COM_t1)[0]
                self.bodies[body_id].H[1, 1] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[1] * (self.particles[f, p].x + self.dt * self.particles[f + 1, p].v - self.bodies[body_id].COM_t1)[1]
                self.bodies[body_id].H[1, 2] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[1] * (self.particles[f, p].x + self.dt * self.particles[f + 1, p].v - self.bodies[body_id].COM_t1)[2]
                self.bodies[body_id].H[2, 0] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[2] * (self.particles[f, p].x + self.dt * self.particles[f + 1, p].v - self.bodies[body_id].COM_t1)[0]
                self.bodies[body_id].H[2, 1] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[2] * (self.particles[f, p].x + self.dt * self.particles[f + 1, p].v - self.bodies[body_id].COM_t1)[1]
                self.bodies[body_id].H[2, 2] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[2] * (self.particles[f, p].x + self.dt * self.particles[f + 1, p].v - self.bodies[body_id].COM_t1)[2]

    @ti.kernel
    def compute_H_svd(self, f: ti.i32):
        for body_id in range(self.n_bodies):
            if self.bodies_i[body_id].mat_cls == MAT_RIGID:
                self.bodies[body_id].U, self.bodies[body_id].S, self.bodies[body_id].V = ti.svd(self.bodies[body_id].H, DTYPE_TI)

    @ti.kernel
    def compute_H_svd_grad(self, f: ti.i32):
        for body_id in range(self.n_bodies):
            if self.bodies_i[body_id].mat_cls == MAT_RIGID:
                self.bodies.grad[body_id].H = self.backward_svd(self.bodies.grad[body_id].U,
                                                                self.bodies.grad[body_id].S,
                                                                self.bodies.grad[body_id].V, self.bodies[body_id].U,
                                                                self.bodies[body_id].S, self.bodies[body_id].V)

    @ti.kernel
    def compute_R(self, f: ti.i32):
        for body_id in range(self.n_bodies):
            if self.bodies_i[body_id].mat_cls == MAT_RIGID:
                self.bodies[body_id].R = self.bodies[body_id].V @ self.bodies[body_id].U.transpose()

    @ti.kernel
    def advect_kernel(self, f: ti.i32):
        for p in range(self.n_particles):
            self.particles[f + 1, p].x = self.particles[f, p].x + self.dt * self.particles[f + 1, p].v

    @ti.kernel
    def advect_kernel_toi(self, f: ti.i32):
        for p in range(self.n_particles):
            self.particles[f + 1, p].x = (self.particles[f, p].x +
                                          self.particles[f, p].delta_x +
                                          self.particles[f, p].toi * self.particles[f + 1, p].pre_collide_v +
                                          (self.dt - self.particles[f, p].toi) * self.particles[f + 1, p].v)

    @ti.kernel
    def advect_kernel_with_rigid(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particles_i[p].mat_cls == MAT_RIGID:  # rigid objects
                body_id = self.particles_i[p].body_id
                self.particles[f + 1, p].x = self.bodies[body_id].R @ (self.particles[f, p].x - self.bodies[body_id].COM_t0) + self.bodies[body_id].COM_t1
            else:  # other particles
                self.particles[f + 1, p].x = self.particles[f, p].x + self.dt * self.particles[f + 1, p].v

    def agent_move(self, f, is_none_action):
        if not is_none_action:
            self.agent.move(f)

    def agent_move_grad(self, f, is_none_action):
        if not is_none_action:
            self.agent.move_grad(f)

    def substep(self, f, is_none_action):
        if self.has_particles:
            self.reset_grid_and_grad(f)

        if self.has_particles:
            self.compute_F_tmp(f)
            self.svd(f)
            self.p2g(f)

        self.agent_move(f, is_none_action)  # this updates agent pos at f + 1

        if self.has_particles:
            self.grid_op(f)
            self.g2p(f)
            self.advect(f)  # this updates particle pos at f + 1

    def substep_grad(self, f, is_none_action):
        if self.has_particles:
            self.advect_grad(f)
            if self.debug_grad:
                print(f'******Grads after advect_grad() at substep {f}, global step {self.cur_step_global}')
                self.print_grads()
            if self.grad_op != 'none':
                self.clip_particle_x_grad(f)
            self.g2p.grad(f)
            if self.debug_grad:
                print(f'******Grads after g2p.grad() at substep {f}, global step {self.cur_step_global}')
                self.print_grads()
            self.grid_op.grad(f)
            if self.debug_grad:
                print(f'******Grads after grid_op.grad() at substep {f}, global step {self.cur_step_global}')
                self.print_grads()
            if self.grad_op != 'none':
                self.clip_grid_grad(f)

        self.agent_move_grad(f, is_none_action)

        if self.has_particles:
            self.p2g.grad(f)
            if self.debug_grad:
                print(f'******Grads after p2g.grad() at substep {f}, global step {self.cur_step_global}')
                self.print_grads()
            self.svd_grad(f)
            if self.debug_grad:
                print(f'******Grads after svd_grad() at substep {f}, global step {self.cur_step_global}')
                self.print_grads()
            self.compute_F_tmp.grad(f)
            if self.debug_grad:
                print(f'******Grads after compute_F_tmp.grad() at substep {f}, global step {self.cur_step_global}')
                self.print_grads()
            if self.grad_op != 'none':
                self.clip_particle_F_grad(f)

    def step(self, action=None):
        if self.grad_enabled:
            if self.cur_substep_local == 0:
                self.actions_buffer = []

        self.step_(action)

        if self.grad_enabled:
            self.actions_buffer.append(action)

        if self.cur_substep_local == 0:
            self.memory_to_cache()

    def step_(self, action=None):
        is_none_action = action is None
        if not is_none_action:
            self.agent.set_action(
                s=self.cur_step_local,
                s_global=self.cur_step_global,
                n_substeps=self.n_substeps,
                action=action
            )

        for i in range(0, self.n_substeps):
            self.substep(self.cur_substep_local, is_none_action)
            self.cur_substep_global += 1

        assert self.cur_substep_global <= self.max_substeps_global

    def step_grad(self, action=None):
        if self.cur_substep_local == 0:
            self.memory_from_cache()

        is_none_action = action is None

        for i in range(self.n_substeps - 1, -1, -1):
            self.cur_substep_global -= 1
            self.substep_grad(self.cur_substep_local, is_none_action)
            if self.log_substep_grad:
                self.log_grads(n=(self.n_substeps * self.trajectory_length - self.cur_substep_global))

        if not is_none_action:
            self.agent.set_action_grad(
                s=self.cur_substep_local // self.n_substeps,
                s_global=self.cur_substep_global // self.n_substeps,
                n_substeps=self.n_substeps,
                action=action
            )
            if self.debug_grad:
                print(f'******Grads after set_action_grad, global step {self.cur_step_global}')
                self.print_grads()
            if self.grad_op != 'none':
                self.clip_action_grad(self.cur_substep_global // self.n_substeps)

    # ------------------------------------ grad operations ------------------------------------- #

    @ti.kernel
    def get_param_grad_kernel(self, particle_param_grad: ti.types.ndarray(), system_param_grad: ti.types.ndarray()):
        for i in ti.static(range(NUM_MATERIAL)):
            particle_param_grad[i, 0] = self.particle_param.grad[i].E
            particle_param_grad[i, 1] = self.particle_param.grad[i].nu
            particle_param_grad[i, 2] = self.particle_param.grad[i].rho
        system_param_grad[0] = self.system_param.grad[None].manipulator_friction
        system_param_grad[1] = self.system_param.grad[None].ground_friction
        system_param_grad[2] = self.system_param.grad[None].container_friction
        system_param_grad[3] = self.system_param.grad[None].yield_stress
        system_param_grad[4] = self.system_param.grad[None].theta_c
        system_param_grad[5] = self.system_param.grad[None].theta_s

    def get_param_grad(self):
        particle_param_grad = np.zeros(shape=(NUM_MATERIAL, 3), dtype=DTYPE_NP)
        system_param_grad = np.zeros(shape=(3,), dtype=DTYPE_NP)
        self.get_param_grad_kernel(particle_param_grad, system_param_grad)

        return {
            'particle_param_grad': particle_param_grad,
            'system_param_grad': system_param_grad
        }

    @ti.kernel
    def reset_particle_grid_grad_records(self):
        self.debug_info[None].min_particle_grad_x = 1e8
        self.debug_info[None].max_particle_grad_x = -1e8
        self.debug_info[None].min_particle_grad_v = 1e8
        self.debug_info[None].max_particle_grad_v = -1e8
        self.debug_info[None].min_particle_grad_C = 1e8
        self.debug_info[None].max_particle_grad_C = -1e8
        self.debug_info[None].min_particle_grad_F = 1e8
        self.debug_info[None].max_particle_grad_F = -1e8
        self.debug_info[None].min_particle_grad_U = 1e8
        self.debug_info[None].max_particle_grad_U = -1e8
        self.debug_info[None].min_particle_grad_V = 1e8
        self.debug_info[None].max_particle_grad_V = -1e8
        self.debug_info[None].min_particle_grad_S = 1e8
        self.debug_info[None].max_particle_grad_S = -1e8
        self.debug_info[None].min_grid_grad_v_in = 1e8
        self.debug_info[None].max_grid_grad_v_in = -1e8
        self.debug_info[None].min_grid_grad_mass = 1e8
        self.debug_info[None].max_grid_grad_mass = -1e8
        self.debug_info[None].min_grid_grad_v_out = 1e8
        self.debug_info[None].max_grid_grad_v_out = -1e8

    @ti.kernel
    def get_min_max_grid_grad(self, f: ti.i32):
        for I in ti.grouped(ti.ndrange(*self.res)):
            ti.atomic_max(self.debug_info[None].max_grid_grad_mass, self.grid.grad[f, I].mass)
            ti.atomic_min(self.debug_info[None].min_grid_grad_mass, self.grid.grad[f, I].mass)
            for i in ti.static(range(self.dim)):
                ti.atomic_max(self.debug_info[None].max_grid_grad_v_in, self.grid.grad[f, I].v_in[i])
                ti.atomic_min(self.debug_info[None].min_grid_grad_v_in, self.grid.grad[f, I].v_in[i])
                ti.atomic_max(self.debug_info[None].max_grid_grad_v_out, self.grid.grad[f, I].v_out[i])
                ti.atomic_min(self.debug_info[None].min_grid_grad_v_out, self.grid.grad[f, I].v_out[i])

    @ti.kernel
    def get_min_max_p_grad(self, f: ti.i32):
        for p in range(self.n_particles):
            for i in ti.static(range(self.dim)):
                ti.atomic_max(self.debug_info[None].max_particle_grad_x, self.particles.grad[f, p].x[i])
                ti.atomic_min(self.debug_info[None].min_particle_grad_x, self.particles.grad[f, p].x[i])
                ti.atomic_max(self.debug_info[None].max_particle_grad_v, self.particles.grad[f, p].v[i])
                ti.atomic_min(self.debug_info[None].min_particle_grad_v, self.particles.grad[f, p].v[i])
                for j in ti.static(range(self.dim)):
                    ti.atomic_max(self.debug_info[None].max_particle_grad_C, self.particles.grad[f, p].C[i, j])
                    ti.atomic_min(self.debug_info[None].min_particle_grad_C, self.particles.grad[f, p].C[i, j])
                    ti.atomic_max(self.debug_info[None].max_particle_grad_F, self.particles.grad[f, p].F[i, j])
                    ti.atomic_min(self.debug_info[None].min_particle_grad_F, self.particles.grad[f, p].F[i, j])
                    ti.atomic_max(self.debug_info[None].max_particle_grad_U, self.particles.grad[f, p].U[i, j])
                    ti.atomic_min(self.debug_info[None].min_particle_grad_U, self.particles.grad[f, p].U[i, j])
                    ti.atomic_max(self.debug_info[None].max_particle_grad_V, self.particles.grad[f, p].V[i, j])
                    ti.atomic_min(self.debug_info[None].min_particle_grad_V, self.particles.grad[f, p].V[i, j])
                    ti.atomic_max(self.debug_info[None].max_particle_grad_S, self.particles.grad[f, p].S[i, j])
                    ti.atomic_min(self.debug_info[None].min_particle_grad_S, self.particles.grad[f, p].S[i, j])

    @ti.kernel
    def clip_particle_F_grad(self, f: ti.i32):
        for p in range(self.n_particles):
            if ti.static(self.grad_op == 'clip'):
                for i in ti.static(range(self.dim)):
                    for j in ti.static(range(self.dim)):
                        F_grad_tmp = ti.min(ti.max(self.particles.grad[f, p].F[i, j], self.grad_min), self.grad_max)
                        self.particles.grad[f, p].F[i, j] = F_grad_tmp
            elif ti.static(self.grad_op == 'normalize'):
                self.particles.grad[f, p].F /= (self.particles.grad[f, p].F.norm() + 1e-6)
            elif ti.static(self.grad_op == 'dynamic-scale'):
                max_grad = ti.max(ti.abs(self.particles.grad[f, p].F))
                max_oom_grad = ti.round(ti.log(max_grad) / ti.log(10))
                delta_oom_grad = max_oom_grad - 4
                self.particles.grad[f, p].F /= (10**delta_oom_grad + 1e-6)

    @ti.kernel
    def clip_particle_x_grad(self, f: ti.i32):
        for p in range(self.n_particles):
            if ti.static(self.grad_op == 'clip'):
                for i in ti.static(range(self.dim)):
                    x_grad_tmp = ti.min(ti.max(self.particles.grad[f, p].x[i], self.grad_min), self.grad_max)
                    self.particles.grad[f, p].x[i] = x_grad_tmp
            elif ti.static(self.grad_op == 'normalize'):
                self.particles.grad[f, p].x /= (self.particles.grad[f, p].x.norm() + 1e-6)
            elif ti.static(self.grad_op == 'dynamic-scale'):
                max_grad = ti.max(ti.abs(self.particles.grad[f, p].x))
                max_oom_grad = ti.round(ti.log(max_grad) / ti.log(10))
                delta_oom_grad = max_oom_grad - 4
                self.particles.grad[f, p].x /= (10**delta_oom_grad + 1e-6)

    @ti.kernel
    def clip_action_grad(self, f: ti.i32):
        if ti.static(self.grad_op == 'clip'):
            for n in ti.static(range(len(self.agent.effectors))):
                for j in ti.static(range(self.agent.effectors[n].action_dim)):
                    self.agent.effectors[n].action_buffer.grad[f][j] = ti.min(ti.max(self.agent.effectors[n].action_buffer.grad[f][j], self.grad_min), self.grad_max)
        elif ti.static(self.grad_op == 'normalize'):
            for n in ti.static(range(len(self.agent.effectors))):
                self.agent.effectors[n].action_buffer.grad[f] /= (self.agent.effectors[n].action_buffer.grad[f].norm() + 1e-6)
        elif ti.static(self.grad_op == 'dynamic-scale'):
            for n in ti.static(range(len(self.agent.effectors))):
                max_grad = ti.max(ti.abs(self.agent.effectors[n].action_buffer.grad[f]))
                max_oom_grad = ti.round(ti.log(max_grad) / ti.log(10))
                delta_oom_grad = max_oom_grad - 4
                self.agent.effectors[n].action_buffer.grad[f] /= (10**delta_oom_grad + 1e-6)

    @ti.kernel
    def clip_grid_grad(self, f: ti.i32):
        for I in ti.grouped(ti.ndrange(*self.res)):
            if ti.static(self.grad_op == 'clip'):
                mass_grad_tmp = ti.min(ti.max(self.grid.grad[f, I].mass, self.grad_min), self.grad_max)
                self.grid.grad[f, I].mass = mass_grad_tmp
                for i in ti.static(range(self.dim)):
                    v_in_grad_tmp = ti.min(ti.max(self.grid.grad[f, I].v_in[i], self.grad_min), self.grad_max)
                    self.grid.grad[f, I].v_in[i] = v_in_grad_tmp
            elif ti.static(self.grad_op == 'normalize'):
                self.grid.grad[f, I].mass /= (ti.abs(self.grid.grad[f, I].mass) + 1e-6)
                self.grid.grad[f, I].v_in /= (self.grid.grad[f, I].v_in.norm() + 1e-6)
            elif ti.static(self.grad_op == 'dynamic-scale'):
                max_grad_mass = ti.max(ti.abs(self.grid.grad[f, I].mass))
                max_oom_grad_mass = ti.round(ti.log(max_grad_mass) / ti.log(10))
                delta_oom_grad_mass = max_oom_grad_mass - 4
                self.grid.grad[f, I].mass /= (10**delta_oom_grad_mass + 1e-6)
                max_grad_v_in = ti.max(ti.abs(self.grid.grad[f, I].v_in))
                max_oom_grad_v_in = ti.round(ti.log(max_grad_v_in) / ti.log(10))
                delta_oom_grad_v_in = max_oom_grad_v_in - 4
                self.grid.grad[f, I].v_in /= (10**delta_oom_grad_v_in + 1e-6)

    def print_particle_grid_grads(self):
        print(f"Max particle grad x: {self.debug_info[None].max_particle_grad_x}")
        print(f"Min particle grad x: {self.debug_info[None].min_particle_grad_x}")
        print(f"Max particle grad v: {self.debug_info[None].max_particle_grad_v}")
        print(f"Min particle grad v: {self.debug_info[None].min_particle_grad_v}")
        print(f"Max particle grad C: {self.debug_info[None].max_particle_grad_C}")
        print(f"Min particle grad C: {self.debug_info[None].min_particle_grad_C}")
        print(f"Max particle grad F: {self.debug_info[None].max_particle_grad_F}")
        print(f"Min particle grad F: {self.debug_info[None].min_particle_grad_F}")
        print(f"Max particle grad U: {self.debug_info[None].max_particle_grad_U}")
        print(f"Min particle grad U: {self.debug_info[None].min_particle_grad_U}")
        print(f"Max particle grad V: {self.debug_info[None].max_particle_grad_V}")
        print(f"Min particle grad V: {self.debug_info[None].min_particle_grad_V}")
        print(f"Max particle grad S: {self.debug_info[None].max_particle_grad_S}")
        print(f"Min particle grad S: {self.debug_info[None].min_particle_grad_S}")
        print(f"Max grid grad v_in: {self.debug_info[None].max_grid_grad_v_in}")
        print(f"Min grid grad v_in: {self.debug_info[None].min_grid_grad_v_in}")
        print(f"Max grid grad mass: {self.debug_info[None].max_grid_grad_mass}")
        print(f"Min grid grad mass: {self.debug_info[None].min_grid_grad_mass}")
        print(f"Max grid grad v_out: {self.debug_info[None].max_grid_grad_v_out}")
        print(f"Min grid grad v_out: {self.debug_info[None].min_grid_grad_v_out}")

    def print_grads(self):
        for j in range(NUM_MATERIAL):
            print(f"Mat {j} Gradient of E: {self.particle_param.grad[j].E}")
            print(f"Mat {j} Gradient of nu: {self.particle_param.grad[j].nu}")
            print(f"Mat {j} Gradient of rho: {self.particle_param.grad[j].rho}")
            print(f"Mat {j} Gradient of mu_temp: {self.particle_param.grad[j].mu_temp}")
            print(f"Mat {j} Gradient of lam_temp: {self.particle_param.grad[j].lam_temp}")
            print(f"Mat {j} Gradient of mu_temp 2: {self.particle_param.grad[j].mu_temp_2}")
            print(f"Mat {j} Gradient of lam_temp 2: {self.particle_param.grad[j].lam_temp_2}")
        print(f"Gradient of sand friction: {self.system_param.grad[None].sand_friction_angle}")
        print(f"Gradient of yield stress: {self.system_param.grad[None].yield_stress}")
        print(f"Gradient of manipulator friction: {self.system_param.grad[None].manipulator_friction}")
        print(f"Gradient of ground friction: {self.system_param.grad[None].ground_friction}")
        print(f"Gradient of container friction: {self.system_param.grad[None].container_friction}")
        print(f"Max epsilon: {self.debug_info[None].max_epsilon}")
        print(f"Min epsilon: {self.debug_info[None].min_epsilon}")
        print(f"Max delta gamma: {self.debug_info[None].max_delta_gamma}")
        print(f"Min delta gamma: {self.debug_info[None].min_delta_gamma}")
        print(f"Max Sigma: {self.debug_info[None].max_Sigma}")
        print(f"Min Sigma: {self.debug_info[None].min_Sigma}")
        print(f"Max F: {self.debug_info[None].max_F}")
        print(f"Min F: {self.debug_info[None].min_F}")
        print(f"Max stress: {self.debug_info[None].max_stress}")
        print(f"Min stress: {self.debug_info[None].min_stress}")
        print(f"Max dStress_dF: {self.debug_info[None].max_dStress_dF}")
        print(f"Min dStress_dF: {self.debug_info[None].min_dStress_dF}")
        print(f"Max dF_dSigma: {self.debug_info[None].max_dF_dSigma}")
        print(f"Min dF_dSigma: {self.debug_info[None].min_dF_dSigma}")
        print(f"Max dStress_dmu: {self.debug_info[None].max_dStress_dmu}")
        print(f"Min dStress_dmu: {self.debug_info[None].min_dStress_dmu}")
        print(f"Max dStress_dlam: {self.debug_info[None].max_dStress_dlam}")
        print(f"Min dStress_dlam: {self.debug_info[None].min_dStress_dlam}")
        print(f"Max dStress_dSigma: {self.debug_info[None].max_dStress_dSigma}")
        print(f"Min dStress_dSigma: {self.debug_info[None].min_dStress_dSigma}")
        print(f"Max dCentre_dSigma: {self.debug_info[None].max_dCentre_dSigma}")
        print(f"Min dCentre_dSigma: {self.debug_info[None].min_dCentre_dSigma}")
        print(f"Min dDelta_gamma_dmu: {self.debug_info[None].min_dDelta_gamma_dmu}")
        print(f"Max dDelta_gamma_dmu: {self.debug_info[None].max_dDelta_gamma_dmu}")
        print(f"Min dDelta_gamma_dlam: {self.debug_info[None].min_dDelta_gamma_dlam}")
        print(f"Max dDelta_gamma_dlam: {self.debug_info[None].max_dDelta_gamma_dlam}")
        print(f"Max dDelta_gamma_dSand_alpha: {self.debug_info[None].max_dDelta_gamma_dSand_alpha}")
        print(f"Min dDelta_gamma_dSand_alpha: {self.debug_info[None].min_dDelta_gamma_dSand_alpha}")
        print(f"Max dSigma_ddeltagamma: {self.debug_info[None].max_dSigma_ddeltagamma}")
        print(f"Min dSigma_ddeltagamma: {self.debug_info[None].min_dSigma_ddeltagamma}")
        print(f"Max dSigma_dmu: {self.debug_info[None].max_dSigma_dmu}")
        print(f"Min dSigma_dmu: {self.debug_info[None].min_dSigma_dmu}")
        print(f"Max dSigma_dlam: {self.debug_info[None].max_dSigma_dlam}")
        print(f"Min dSigma_dlam: {self.debug_info[None].min_dSigma_dlam}")
        print(f"Max dSigma_dSand_alpha: {self.debug_info[None].max_dSigma_dSand_alpha}")
        print(f"Min dSigma_dSand_alpha: {self.debug_info[None].min_dSigma_dSand_alpha}")

        self.reset_particle_grid_grad_records()
        self.get_min_max_p_grad(self.cur_substep_local)
        self.get_min_max_grid_grad(self.cur_substep_local)
        self.print_particle_grid_grads()

        trajectory_grads = self.agent.get_grad(n=400)
        print(f"Max action grad: {np.max(trajectory_grads)}")
        print(f"Min action grad: {np.min(trajectory_grads)}")

        input("Press Enter to continue...")

    def log_grads(self, n):
        if self.tb_logger is None:
            pass
        else:
            self.reset_particle_grid_grad_records()
            self.get_min_max_p_grad(self.cur_substep_local)
            self.get_min_max_grid_grad(self.cur_substep_local)

            self.tb_logger.add_scalar('Grad-p/x_min', self.debug_info[None].min_particle_grad_x, n)
            self.tb_logger.add_scalar('Grad-p/x_max', self.debug_info[None].max_particle_grad_x, n)
            self.tb_logger.add_scalar('Grad-p/v_min', self.debug_info[None].min_particle_grad_v, n)
            self.tb_logger.add_scalar('Grad-p/v_max', self.debug_info[None].max_particle_grad_v, n)
            self.tb_logger.add_scalar('Grad-p/F_min', self.debug_info[None].min_particle_grad_F, n)
            self.tb_logger.add_scalar('Grad-p/F_max', self.debug_info[None].max_particle_grad_F, n)
            self.tb_logger.add_scalar('Grad-grid/mass_min', self.debug_info[None].min_grid_grad_mass, n)
            self.tb_logger.add_scalar('Grad-grid/mass_max', self.debug_info[None].max_grid_grad_mass, n)
            self.tb_logger.add_scalar('Grad-grid/v_in_min', self.debug_info[None].min_grid_grad_v_in, n)
            self.tb_logger.add_scalar('Grad-grid/v_in_max', self.debug_info[None].max_grid_grad_v_in, n)
            self.tb_logger.add_scalar('Grad-grid/v_out_min', self.debug_info[None].min_grid_grad_v_out, n)
            self.tb_logger.add_scalar('Grad-grid/v_out_max', self.debug_info[None].max_grid_grad_v_out, n)

            self.tb_logger.add_scalar('Grad_param/E', self.particle_param.grad[SAND].E, n)
            self.tb_logger.add_scalar('Grad_param/nu', self.particle_param.grad[SAND].nu, n)
            self.tb_logger.add_scalar('Grad_param/rho', self.particle_param.grad[SAND].rho, n)
            self.tb_logger.add_scalar('Grad_param/sand_angle', self.system_param.grad[None].sand_friction_angle, n)

            trajectory_grads = self.agent.get_grad(self.trajectory_length)
            self.tb_logger.add_scalar('Grad_action/0_max', np.max(trajectory_grads[:, 0]), n)
            self.tb_logger.add_scalar('Grad_action/0_min', np.min(trajectory_grads[:, 0]), n)
            self.tb_logger.add_scalar('Grad_action/1_max', np.max(trajectory_grads[:, 1]), n)
            self.tb_logger.add_scalar('Grad_action/1_min', np.min(trajectory_grads[:, 1]), n)
            self.tb_logger.add_scalar('Grad_action/2_max', np.max(trajectory_grads[:, 2]), n)
            self.tb_logger.add_scalar('Grad_action/2_min', np.min(trajectory_grads[:, 2]), n)
            self.tb_logger.add_scalar('Grad_action/3_max', np.max(trajectory_grads[:, 3]), n)
            self.tb_logger.add_scalar('Grad_action/3_min', np.min(trajectory_grads[:, 3]), n)
            self.tb_logger.add_scalar('Grad_action/4_max', np.max(trajectory_grads[:, 4]), n)
            self.tb_logger.add_scalar('Grad_action/4_min', np.min(trajectory_grads[:, 4]), n)
            self.tb_logger.add_scalar('Grad_action/5_max', np.max(trajectory_grads[:, 5]), n)
            self.tb_logger.add_scalar('Grad_action/5_min', np.min(trajectory_grads[:, 5]), n)

    # ------------------------------------ io -------------------------------------#
    @ti.kernel
    def readframe(self, f: ti.i32, x: ti.types.ndarray(), v: ti.types.ndarray(), C: ti.types.ndarray(),
                  F: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                x[i, j] = self.particles[f, i].x[j]
                v[i, j] = self.particles[f, i].v[j]
                for k in ti.static(range(self.dim)):
                    C[i, j, k] = self.particles[f, i].C[j, k]
                    F[i, j, k] = self.particles[f, i].F[j, k]

    @ti.kernel
    def setframe(self, f: ti.i32, x: ti.types.ndarray(), v: ti.types.ndarray(), C: ti.types.ndarray(),
                 F: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                self.particles[f, i].x[j] = x[i, j]
                self.particles[f, i].v[j] = v[i, j]
                for k in ti.static(range(self.dim)):
                    self.particles[f, i].C[j, k] = C[i, j, k]
                    self.particles[f, i].F[j, k] = F[i, j, k]

    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        for i in range(self.n_particles):
            self.particles[target, i].x = self.particles[source, i].x
            self.particles[target, i].v = self.particles[source, i].v
            self.particles[target, i].F = self.particles[source, i].F
            self.particles[target, i].C = self.particles[source, i].C

    @ti.kernel
    def copy_grad(self, source: ti.i32, target: ti.i32):
        for i in range(self.n_particles):
            self.particles.grad[target, i].x = self.particles.grad[source, i].x
            self.particles.grad[target, i].v = self.particles.grad[source, i].v
            self.particles.grad[target, i].F = self.particles.grad[source, i].F
            self.particles.grad[target, i].C = self.particles.grad[source, i].C

    @ti.kernel
    def reset_grad_till_frame(self, f: ti.i32):
        for i, j in ti.ndrange(f, self.n_particles):
            self.particles.grad[i, j].x.fill(0)
            self.particles.grad[i, j].v.fill(0)
            self.particles.grad[i, j].C.fill(0)
            self.particles.grad[i, j].F.fill(0)
            self.particles.grad[i, j].F_tmp.fill(0)
            self.particles.grad[i, j].U.fill(0)
            self.particles.grad[i, j].V.fill(0)
            self.particles.grad[i, j].S.fill(0)

    def get_state(self):
        f = self.cur_substep_local
        s = self.cur_step_local

        state = {}

        if self.has_particles:
            state['x'] = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
            state['v'] = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
            state['C'] = np.zeros((self.n_particles, self.dim, self.dim), dtype=DTYPE_NP)
            state['F'] = np.zeros((self.n_particles, self.dim, self.dim), dtype=DTYPE_NP)
            self.readframe(f, state['x'], state['v'], state['C'], state['F'])

        if self.has_agent:
            state['agent'] = self.agent.get_state(f)

        return state

    def set_state(self, f_global, state):
        f = self.f_global_to_f_local(f_global)
        s = self.f_global_to_s_local(f_global)

        if self.has_particles:
            self.setframe(f, state['x'], state['v'], state['C'], state['F'])

        if self.has_agent:
            self.agent.set_state(f, state['agent'])

    @ti.kernel
    def get_x_kernel(self, f: ti.i32, x: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                x[i, j] = self.particles[f, i].x[j]

    def get_x(self, f=None):
        if f is None:
            f = self.cur_substep_local

        x = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
        if self.has_particles:
            self.get_x_kernel(f, x)
        return x

    @ti.kernel
    def get_state_RL_kernel(self, f: ti.i32, x: ti.types.ndarray(), v: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                x[i, j] = self.particles[f, i].x[j]
                v[i, j] = self.particles[f, i].v[j]

    def get_state_RL(self):
        f = self.cur_substep_local
        s = self.cur_step_local
        state = {}
        if self.has_particles:
            state['x'] = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
            state['v'] = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
            self.get_state_RL_kernel(f, state['x'], state['v'])
        if self.has_agent:
            state['agent'] = self.agent.get_state(f)
        return state

    @ti.kernel
    def get_state_render_kernel(self, f: ti.i32):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                self.particles_render[i].x[j] = ti.cast(self.particles[f, i].x[j], DTYPE_TI)
            self.particles_render[i].radius = ti.cast(self.bodies_i[self.particles_i[i].body_id].p_radius, DTYPE_TI)

    def get_state_render(self, f):
        self.get_state_render_kernel(f)
        return self.particles_render

    @ti.kernel
    def get_v_kernel(self, f: ti.i32, v: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                v[i, j] = self.particles[f, i].v[j]

    def get_v(self, f):
        v = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
        if self.has_particles:
            self.get_v_kernel(f, v)
        return v

    def memory_to_cache(self):
        if self.grad_enabled:
            ckpt_start_step = self.cur_substep_global - self.max_substeps_local
            ckpt_end_step = self.cur_substep_global - 1
            ckpt_name = f'{ckpt_start_step:06d}'

            if self.ckpt_dest == 'disk':
                ckpt = {}
                if self.has_particles:
                    self.readframe(0, self.x_np, self.v_np, self.C_np, self.F_np)
                    ckpt['x'] = self.x_np
                    ckpt['v'] = self.v_np
                    ckpt['C'] = self.C_np
                    ckpt['F'] = self.F_np
                    ckpt['actions'] = self.actions_buffer

                if self.has_agent:
                    ckpt['agent'] = self.agent.get_ckpt()

                # save to /tmp
                ckpt_file = os.path.join(self.ckpt_dir, f'{ckpt_name}.pkl')
                if os.path.exists(ckpt_file):
                    os.remove(ckpt_file)
                pkl.dump(ckpt, open(ckpt_file, 'wb'))

            elif self.ckpt_dest in ['cpu', 'gpu']:
                if ckpt_name not in self.ckpt_ram:
                    self.ckpt_ram[ckpt_name] = {}

                    if self.ckpt_dest == 'cpu':
                        device = 'cpu'
                    elif self.ckpt_dest == 'gpu':
                        device = 'cuda'
                    if self.has_particles:
                        self.ckpt_ram[ckpt_name]['x'] = torch.zeros((self.n_particles, self.dim), dtype=DTYPE_TC,
                                                                    device=device)
                        self.ckpt_ram[ckpt_name]['v'] = torch.zeros((self.n_particles, self.dim), dtype=DTYPE_TC,
                                                                    device=device)
                        self.ckpt_ram[ckpt_name]['C'] = torch.zeros((self.n_particles, self.dim, self.dim),
                                                                    dtype=DTYPE_TC, device=device)
                        self.ckpt_ram[ckpt_name]['F'] = torch.zeros((self.n_particles, self.dim, self.dim),
                                                                    dtype=DTYPE_TC, device=device)

                if self.has_particles:
                    self.readframe(
                        0,
                        self.ckpt_ram[ckpt_name]['x'],
                        self.ckpt_ram[ckpt_name]['v'],
                        self.ckpt_ram[ckpt_name]['C'],
                        self.ckpt_ram[ckpt_name]['F'],
                    )

                self.ckpt_ram[ckpt_name]['actions'] = list(self.actions_buffer)

                if self.has_agent:
                    self.agent.get_ckpt(ckpt_name)

            else:
                assert False

            # print(f'[Forward] Cached step {ckpt_start_step} to {ckpt_end_step}. {t2-t1:.2f}s {t3-t2:.2f}s')

        # restart from frame 0 in memory
        if self.has_particles:
            self.copy_frame(self.max_substeps_local, 0)

        if self.has_agent:
            self.agent.copy_frame(self.max_substeps_local, 0)

        # print(f'[Forward] Memory refreshed. Now starts from global step {self.cur_substep_global}.')

    def memory_from_cache(self):
        assert self.grad_enabled
        if self.has_particles:
            self.copy_frame(0, self.max_substeps_local)
            self.copy_grad(0, self.max_substeps_local)
            self.reset_grad_till_frame(self.max_substeps_local)

        if self.has_agent:
            self.agent.copy_frame(0, self.max_substeps_local)
            self.agent.copy_grad(0, self.max_substeps_local)
            self.agent.reset_grad_till_frame(self.max_substeps_local)

        ckpt_start_step = self.cur_substep_global - self.max_substeps_local
        ckpt_end_step = self.cur_substep_global - 1
        ckpt_name = f'{ckpt_start_step:06d}'

        if self.ckpt_dest == 'disk':
            ckpt_file = os.path.join(self.ckpt_dir, f'{ckpt_start_step:06d}.pkl')
            assert os.path.exists(ckpt_file)
            ckpt = pkl.load(open(ckpt_file, 'rb'))

            if self.has_particles:
                self.setframe(0, ckpt['x'], ckpt['v'], ckpt['C'], ckpt['F'])

            if self.has_agent:
                self.agent.set_ckpt(ckpt=ckpt['agent'])

        elif self.ckpt_dest in ['cpu', 'gpu']:
            if self.has_particles:
                ckpt = self.ckpt_ram[ckpt_name]
                self.setframe(0, ckpt['x'], ckpt['v'], ckpt['C'], ckpt['F'])

            if self.has_agent:
                self.agent.set_ckpt(ckpt_name=ckpt_name)

        else:
            assert False

        # now that we loaded the first frame, we do a forward pass to fill up the rest 
        self.cur_substep_global = ckpt_start_step
        for action in ckpt['actions']:
            self.step_(action)

        # print(f'[Backward] Loading step {ckpt_start_step} to {ckpt_end_step} from cache. {t2-t1:.2f}s {t3-t2:.2f}s {t4-t3:.2f}s')
        # print(f'[Backward] Memory reloaded. Now starts from global step {ckpt_start_step}.')

    def clear_ckpt(self):
        if self.ckpt_dest == 'disk':
            for filename in os.listdir(self.ckpt_dir):
                if os.path.isfile(os.path.join(self.ckpt_dir, filename)):
                    os.remove(os.path.join(self.ckpt_dir, filename))
            os.rmdir(self.ckpt_dir)
        elif self.ckpt_dest in ['cpu', 'gpu']:
            self.ckpt_ram = {}
            if self.has_agent:
                self.agent.clear_ckpt()
