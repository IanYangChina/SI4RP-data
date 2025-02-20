import taichi as ti
from doma.engine.configs.macros import DTYPE_TI


@ti.data_oriented
class Loss:
    def __init__(self, max_loss_steps, weight=None, target_file=None, logger=None):
        self.weight = weight
        self.target_file = target_file
        self.inf = 1e8
        self.max_loss_steps = max_loss_steps

        self.step_loss = ti.field(dtype=DTYPE_TI, shape=(self.max_loss_steps,), needs_grad=True)
        self.total_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)

        self.logger = logger

    def build(self, sim):
        self.sim = sim
        self.res = sim.res
        self.n_grid = sim.n_grid
        self.dx = sim.dx
        self.dim = sim.dim

        if self.sim.agent is not None:
            self.agent = sim.agent

        if self.sim.particles is not None:
            self.particle_x = sim.particles.x
            self.particle_mat = sim.particles_i.mat
            self.n_particles = sim.n_particles
            self.grid_mass = sim.grid.mass

        if self.target_file is not None:
            self.load_target(self.target_file)

        self.reset()

    def reset_grad(self):
        self.step_loss.grad.fill(1)
        self.total_loss.grad.fill(1)

    def load_target(self, path):
        pass

    @ti.kernel
    def clear_loss(self):
        self.step_loss.fill(0)
        self.step_loss.grad.fill(0)

        self.total_loss.fill(0)
        self.total_loss.grad.fill(1)

    @ti.kernel
    def clear_losses(self):
        pass

    def reset(self):
        self.clear_loss()
        self.clear_losses()

    def step(self):
        # compute loss for step self.sim.cur_step_global-1
        self.compute_step_loss(self.sim.cur_step_global - 1, self.sim.cur_substep_local)

    def step_grad(self):
        # compute loss for step self.sim.cur_step_global-1
        self.compute_step_loss_grad(self.sim.cur_step_global - 1, self.sim.cur_substep_local)

    def get_step_loss(self, s):
        pass