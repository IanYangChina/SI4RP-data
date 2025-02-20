import taichi as ti
from doma.engine.configs.macros import DTYPE_TI, DTYPE_NP
from .loss import Loss
import open3d as o3d
import numpy as np
from scipy.optimize import linear_sum_assignment
from doma.engine.utils.mesh_ops import generate_particles_from_mesh


@ti.data_oriented
class DPSIPlanningLosses(Loss):
    def __init__(
            self,
            matching_mat,
            target_pcd_height_map_path,
            height_map_res,
            height_map_size,
            **kwargs,
    ):
        super(DPSIPlanningLosses, self).__init__(**kwargs)
        self.matching_mat = matching_mat
        self.target_pcd_height_map_path = target_pcd_height_map_path
        self.height_map_res = height_map_res
        self.height_map_size = height_map_size * 1000  # mm
        self.height_map_xy_offset = (0.25 * 1000, 0.25 * 1000)  # centre point of the height map
        self.height_map_pixel_size = self.height_map_size / self.height_map_res

        self.is_nan_particle = 0

    def build(self, sim):
        self.n_particles_matching_mat = sim.n_particles_per_mat[self.matching_mat]

        self.height_map_loss_pcd = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.height_map = ti.field(dtype=DTYPE_TI, shape=(self.height_map_res, self.height_map_res), needs_grad=True)

        self.height_map_pcd_target = ti.field(dtype=DTYPE_TI, shape=(self.height_map_res, self.height_map_res),
                                              needs_grad=False)

        height_map_pcd_target_np = np.load(self.target_pcd_height_map_path).astype(DTYPE_NP)
        self.height_map_pcd_target.from_numpy(height_map_pcd_target_np)
        del height_map_pcd_target_np

        super(DPSIPlanningLosses, self).build(sim)

    def reset_grad(self):
        super(DPSIPlanningLosses, self).reset_grad()

        self.height_map_loss_pcd.grad.fill(0)
        self.height_map.grad.fill(0)

    @ti.func
    def from_xy_to_uv(self, x: DTYPE_TI, y: DTYPE_TI):
        # this does not need to be differentiable as the loss is connected to the z values of the particles/points
        u = (x - self.height_map_xy_offset[0]) / self.height_map_pixel_size + self.height_map_res / 2
        v = (y - self.height_map_xy_offset[1]) / self.height_map_pixel_size + self.height_map_res / 2
        return ti.floor(u, ti.i32), ti.floor(v, ti.i32)

    @ti.kernel
    def clear_losses(self):
        self.height_map_loss_pcd.fill(0)
        self.height_map_loss_pcd.grad.fill(0)
        self.height_map.fill(0)
        self.height_map.grad.fill(0)

    def compute_step_loss(self, s, f):
        pass

    def compute_step_loss_grad(self, s, f):
        pass

    """Height map loss"""
    def compute_height_map_loss(self, f):
        self.calculate_height_map_sim_particles(f)
        self.compute_height_map_euclidean_distance()

    def compute_height_map_loss_grad(self, f):
        self.compute_height_map_euclidean_distance.grad()
        self.calculate_height_map_sim_particles.grad(f)

    @ti.kernel
    def calculate_height_map_sim_particles(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particle_mat[p] == self.matching_mat:
                u, v = self.from_xy_to_uv(self.particle_x[f, p][0] * 1000, self.particle_x[f, p][1] * 1000)
                ti.atomic_max(self.height_map[u, v], (self.particle_x[f, p][2] * 1000))

    @ti.kernel
    def compute_height_map_euclidean_distance(self):
        for i, j in self.height_map_pcd_target:
            d = ti.sqrt((self.height_map_pcd_target[i, j] - self.height_map[i, j]) ** 2)
            self.height_map_loss_pcd[None] += d

    @ti.kernel
    def get_final_loss_kernel(self):
        self.total_loss[None] += self.height_map_loss_pcd[None]

    def validate_nan_inf_particles(self, f):
        particles = self.particle_x.to_numpy()[f]
        if np.any(np.isnan(particles)) or np.any(np.isinf(particles)):
            return True
        else:
            return False

    def get_final_loss(self):
        particle_has_naninf = self.validate_nan_inf_particles(self.sim.cur_substep_local)
        self.compute_height_map_loss(self.sim.cur_substep_local)
        self.get_final_loss_kernel()

        loss_info = {
            'particle_has_naninf': particle_has_naninf,
            'final_height_map': self.height_map.to_numpy(),
            'height_map_loss_pcd': self.height_map_loss_pcd[None],
            'total_loss': self.total_loss[None],
        }
        return loss_info

    def get_final_loss_grad(self):
        self.get_final_loss_kernel.grad()
        self.compute_height_map_loss_grad(self.sim.cur_substep_local)
