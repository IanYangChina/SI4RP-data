import taichi as ti
from doma.engine.configs.macros import DTYPE_TI, DTYPE_NP
from .loss import Loss
import open3d as o3d
import numpy as np
from scipy.optimize import linear_sum_assignment
from doma.engine.utils.mesh_ops import generate_particles_from_mesh


@ti.data_oriented
class DPSILosses(Loss):
    def __init__(
            self,
            matching_mat,
            exponential_distance,
            averaging_loss,
            point_distance_rs_loss,
            point_distance_sr_loss,
            target_pcd_path,
            target_pcd_offset,
            down_sample_voxel_size,
            particle_distance_rs_loss,
            particle_distance_sr_loss,
            target_mesh_path,
            voxelize_res,
            target_mesh_start_pos,
            target_mesh_offset,
            particle_density,
            load_height_map,
            target_pcd_height_map_path,
            height_map_loss,
            height_map_res,
            height_map_size,
            emd_point_distance_loss,
            emd_particle_distance_loss,
            **kwargs,
    ):
        super(DPSILosses, self).__init__(**kwargs)
        self.matching_mat = matching_mat
        self.exponential_distance = exponential_distance
        self.averaging_loss = averaging_loss

        self.point_distance_rs_loss = point_distance_rs_loss
        self.point_distance_sr_loss = point_distance_sr_loss
        self.target_pcd_path = target_pcd_path
        self.target_pcd_offset = np.array(target_pcd_offset).astype(DTYPE_NP)
        self.down_sample_voxel_size = down_sample_voxel_size

        self.particle_distance_rs_loss = particle_distance_rs_loss
        self.particle_distance_sr_loss = particle_distance_sr_loss
        self.target_mesh_path = target_mesh_path
        self.voxelize_res = voxelize_res
        self.target_mesh_start_pos = np.array(target_mesh_start_pos).astype(DTYPE_NP)
        self.target_mesh_offset = np.array(target_mesh_offset).astype(DTYPE_NP)
        self.particle_density = particle_density

        self.load_height_map = load_height_map
        self.target_pcd_height_map_path = target_pcd_height_map_path
        self.height_map_loss = height_map_loss
        self.height_map_res = height_map_res
        self.height_map_size = height_map_size * 1000  # mm
        self.height_map_xy_offset = (0.25 * 1000, 0.25 * 1000)  # centre point of the height map
        self.height_map_pixel_size = self.height_map_size / self.height_map_res

        self.pcd_linear_assignment_failed = False
        self.emd_point_distance_loss = emd_point_distance_loss
        self.particle_linear_assignment_failed = False
        self.emd_particle_distance_loss = emd_particle_distance_loss

        self.is_nan_particle = 0

    def build(self, sim):
        self.n_particles_matching_mat = sim.n_particles_per_mat[self.matching_mat]

        target_pcd = o3d.io.read_point_cloud(self.target_pcd_path).voxel_down_sample(
            voxel_size=self.down_sample_voxel_size)
        self.target_pcd_points_np = np.asarray(target_pcd.points, dtype=DTYPE_NP) + self.target_pcd_offset
        self.target_pcd_points_np *= 1000  # convert to mm
        self.n_target_pcd_points = self.target_pcd_points_np.shape[0]
        self.target_pcd_points = ti.Vector.field(3, dtype=DTYPE_TI,
                                                 shape=self.n_target_pcd_points, needs_grad=False)
        self.target_pcd_points.from_numpy(self.target_pcd_points_np)

        self.chamfer_loss_pcd = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.avg_point_distance_sr = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.avg_point_distance_rs = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.point_distance_rs_ind_pairs = ti.field(dtype=ti.i32, shape=self.n_target_pcd_points, needs_grad=False)
        self.point_distance_sr_ind_pairs = ti.field(dtype=ti.i32, shape=sim.n_particles, needs_grad=False)

        # due to the linear assignment algorithm
        # num. of target particles should be smaller than num. of simulated particles for computing EMD loss
        done = False
        ptcl_d = self.particle_density
        while not done:
            target_particles_from_mesh_np = generate_particles_from_mesh(file=self.target_mesh_path,
                                                                         voxelize_res=self.voxelize_res,
                                                                         particle_density=ptcl_d,
                                                                         pos=self.target_mesh_start_pos)
            if target_particles_from_mesh_np.shape[0] < self.n_particles_matching_mat:
                done = True
            else:
                ptcl_d *= 0.95

        target_particles_from_mesh_np += self.target_mesh_offset
        target_particles_from_mesh_np *= 1000  # convert to mm
        self.n_target_particles_from_mesh = target_particles_from_mesh_np.shape[0]
        self.target_particles_from_mesh = ti.Vector.field(3, dtype=DTYPE_TI,
                                                          shape=self.n_target_particles_from_mesh, needs_grad=False)
        self.target_particles_from_mesh.from_numpy(target_particles_from_mesh_np.astype(DTYPE_NP))
        self.chamfer_loss_particle = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.avg_particle_distance_sr = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.avg_particle_distance_rs = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.particle_distance_rs_ind_pairs = ti.field(dtype=ti.i32, shape=self.n_target_particles_from_mesh, needs_grad=False)
        self.particle_distance_sr_ind_pairs = ti.field(dtype=ti.i32, shape=sim.n_particles, needs_grad=False)
        del target_particles_from_mesh_np

        self.height_map_loss_pcd = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.height_map = ti.field(dtype=DTYPE_TI, shape=(self.height_map_res, self.height_map_res), needs_grad=True)

        self.height_map_pcd_target = ti.field(dtype=DTYPE_TI, shape=(self.height_map_res, self.height_map_res),
                                              needs_grad=False)
        if not self.load_height_map:
            self.height_map_pcd_target.fill(0)
            self.calculate_height_map_pcd()
        else:
            height_map_pcd_target_np = np.load(self.target_pcd_height_map_path).astype(DTYPE_NP)
            self.height_map_pcd_target.from_numpy(height_map_pcd_target_np)
            del height_map_pcd_target_np

        self.emd_point_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.emd_particle_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.emd_point_ind_pairs = ti.Vector.field(2, dtype=ti.i32, shape=self.n_target_pcd_points, needs_grad=False)
        self.emd_particle_ind_pairs = ti.Vector.field(2, dtype=ti.i32, shape=self.n_target_particles_from_mesh, needs_grad=False)
        self.emd_point_distance_matrix = ti.field(dtype=DTYPE_TI,
                                                  shape=(self.n_target_pcd_points, self.n_particles_matching_mat),
                                                  needs_grad=False)
        self.emd_particle_distance_matrix = ti.field(dtype=DTYPE_TI,
                                                     shape=(self.n_target_particles_from_mesh, self.n_particles_matching_mat),
                                                     needs_grad=False)

        super(DPSILosses, self).build(sim)

    def reset_grad(self):
        super(DPSILosses, self).reset_grad()
        self.chamfer_loss_pcd.grad.fill(0)
        self.chamfer_loss_particle.grad.fill(0)
        self.avg_point_distance_rs.grad.fill(0)
        self.avg_point_distance_sr.grad.fill(0)
        self.avg_particle_distance_rs.grad.fill(0)
        self.avg_particle_distance_sr.grad.fill(0)

        self.height_map_loss_pcd.grad.fill(0)
        self.height_map.grad.fill(0)

        self.emd_point_loss.grad.fill(0)
        self.emd_particle_loss.grad.fill(0)

    @ti.func
    def from_xy_to_uv(self, x: DTYPE_TI, y: DTYPE_TI):
        # this does not need to be differentiable as the loss is connected to the z values of the particles/points
        u = (x - self.height_map_xy_offset[0]) / self.height_map_pixel_size + self.height_map_res / 2
        v = (y - self.height_map_xy_offset[1]) / self.height_map_pixel_size + self.height_map_res / 2
        return ti.floor(u, ti.i32), ti.floor(v, ti.i32)

    @ti.kernel
    def calculate_height_map_pcd(self):
        for i in range(self.n_target_pcd_points):
            u, v = self.from_xy_to_uv(self.target_pcd_points[i][0], self.target_pcd_points[i][1])
            ti.atomic_max(self.height_map_pcd_target[u, v], self.target_pcd_points[i][2])

    @ti.kernel
    def clear_losses(self):
        self.chamfer_loss_pcd.fill(0)
        self.chamfer_loss_pcd.grad.fill(0)
        self.chamfer_loss_particle.fill(0)
        self.chamfer_loss_particle.grad.fill(0)

        self.point_distance_sr_ind_pairs.fill(-1)
        self.point_distance_rs_ind_pairs.fill(-1)
        self.avg_point_distance_rs.fill(0)
        self.avg_point_distance_rs.grad.fill(0)
        self.avg_point_distance_sr.fill(0)
        self.avg_point_distance_sr.grad.fill(0)

        self.particle_distance_sr_ind_pairs.fill(-1)
        self.particle_distance_rs_ind_pairs.fill(-1)
        self.avg_particle_distance_rs.fill(0)
        self.avg_particle_distance_rs.grad.fill(0)
        self.avg_particle_distance_sr.fill(0)
        self.avg_particle_distance_sr.grad.fill(0)

        self.height_map_loss_pcd.fill(0)
        self.height_map_loss_pcd.grad.fill(0)
        self.height_map.fill(0)
        self.height_map.grad.fill(0)

        self.emd_point_ind_pairs.fill(-1)
        self.emd_particle_ind_pairs.fill(-1)
        self.emd_point_loss.fill(0)
        self.emd_point_loss.grad.fill(0)
        self.emd_particle_loss.fill(0)
        self.emd_particle_loss.grad.fill(0)

    def compute_step_loss(self, s, f):
        pass

    def compute_step_loss_grad(self, s, f):
        pass

    """Height map loss"""
    def compute_height_map_loss(self, f):
        self.calculate_height_map_sim_particles(f)
        if not self.exponential_distance:
            self.compute_height_map_euclidean_distance()
        else:
            self.compute_height_map_exponential_distance()

    def compute_height_map_loss_grad(self, f):
        if not self.exponential_distance:
            self.compute_height_map_euclidean_distance.grad()
        else:
            self.compute_height_map_exponential_distance.grad()
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
    def compute_height_map_exponential_distance(self):
        for i, j in self.height_map_pcd_target:
            d = 40 * (1 - ti.exp(-0.1 * ti.sqrt((self.height_map_pcd_target[i, j] - self.height_map[i, j]) ** 2)))
            self.height_map_loss_pcd[None] += d

    @ti.func
    def arcosh(self, x):
        return ti.log(x + ti.sqrt(x ** 2 - 1))

    @ti.kernel
    def compute_height_map_hyperbolic_distance(self):
        for i, j in self.height_map_pcd_target:
            d = 200 * self.arcosh(1 + (self.height_map_pcd_target[i, j] - self.height_map[i, j]) ** 2)
            self.height_map_loss_pcd[None] += d

    """Chamfer loss in mm"""
    @ti.func
    def compute_euclidean_distance(self, a, b):
        return ti.sqrt(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2))

    @ti.func
    def compute_hyperbolic_distance(self, a, b):
        return 200 * self.arcosh(1 + (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    @ti.func
    def compute_exponential_euclidean_distance(self, a, b):
        # in mm
        return 40 * (1 - ti.exp(-0.1 * ti.sqrt(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2))))

    """Point exponential euclidean distance loss"""
    @ti.kernel
    def compute_point_exponential_distance_rs_ind_pairs(self, f: ti.i32):
        for i in range(self.n_target_pcd_points):
            smallest_distance: DTYPE_TI = 100
            for p in range(self.n_particles):
                if self.particle_mat[p] == self.matching_mat:
                    d = self.compute_exponential_euclidean_distance(self.particle_x[f, p] * 1000,
                                                                    self.target_pcd_points[i])
                    if d < smallest_distance:
                        smallest_distance = d
                        self.point_distance_rs_ind_pairs[i] = p

    @ti.kernel
    def compute_point_exponential_distance_rs_kernel(self, f: ti.i32):
        for i in range(self.n_target_pcd_points):
            p = self.point_distance_rs_ind_pairs[i]
            d = self.compute_exponential_euclidean_distance(self.particle_x[f, p] * 1000,
                                                            self.target_pcd_points[i])
            self.avg_point_distance_rs[None] += d

    @ti.kernel
    def compute_point_exponential_distance_sr_ind_pairs(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particle_mat[p] == self.matching_mat:
                smallest_distance: DTYPE_TI = 100
                for i in range(self.n_target_pcd_points):
                    d = self.compute_exponential_euclidean_distance(self.particle_x[f, p] * 1000,
                                                        self.target_pcd_points[i])
                    if d < smallest_distance:
                        smallest_distance = d
                        self.point_distance_sr_ind_pairs[p] = i

    @ti.kernel
    def compute_point_exponential_distance_sr_kernel(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particle_mat[p] == self.matching_mat:
                i = self.point_distance_sr_ind_pairs[p]
                d = self.compute_exponential_euclidean_distance(self.particle_x[f, p] * 1000,
                                                                self.target_pcd_points[i])
                self.avg_point_distance_sr[None] += d

    """Particle exponential euclidean distance loss"""
    @ti.kernel
    def compute_particle_exponential_distance_sr_ind_pairs(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particle_mat[p] == self.matching_mat:
                smallest_distance: DTYPE_TI = 100
                for i in range(self.n_target_particles_from_mesh):
                    d = self.compute_exponential_euclidean_distance(self.particle_x[f, p] * 1000,
                                                        self.target_particles_from_mesh[i])
                    if d < smallest_distance:
                        smallest_distance = d
                        self.particle_distance_sr_ind_pairs[p] = i

    @ti.kernel
    def compute_particle_exponential_distance_sr_kernel(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particle_mat[p] == self.matching_mat:
                i = self.particle_distance_sr_ind_pairs[p]
                d = self.compute_exponential_euclidean_distance(self.particle_x[f, p] * 1000,
                                                                self.target_particles_from_mesh[i])
                self.avg_particle_distance_sr[None] += d

    @ti.kernel
    def compute_particle_exponential_distance_rs_ind_pairs(self, f: ti.i32):
        for i in range(self.n_target_particles_from_mesh):
            smallest_distance: DTYPE_TI = 100
            for p in range(self.n_particles):
                if self.particle_mat[p] == self.matching_mat:
                    d = self.compute_exponential_euclidean_distance(self.particle_x[f, p] * 1000,
                                                        self.target_particles_from_mesh[i])
                    if d < smallest_distance:
                        smallest_distance = d
                        self.particle_distance_rs_ind_pairs[i] = p

    @ti.kernel
    def compute_particle_exponential_distance_rs_kernel(self, f: ti.i32):
        for i in range(self.n_target_particles_from_mesh):
            p = self.particle_distance_rs_ind_pairs[i]
            d = self.compute_exponential_euclidean_distance(self.particle_x[f, p] * 1000,
                                                            self.target_particles_from_mesh[i])
            self.avg_particle_distance_rs[None] += d

    """Point euclidean distance loss"""
    @ti.kernel
    def compute_point_euclidean_distance_rs_ind_pairs(self, f: ti.i32):
        for i in range(self.n_target_pcd_points):
            smallest_distance: DTYPE_TI = 100
            for p in range(self.n_particles):
                if self.particle_mat[p] == self.matching_mat:
                    d = self.compute_euclidean_distance(self.particle_x[f, p] * 1000,
                                                        self.target_pcd_points[i])
                    if d < smallest_distance:
                        smallest_distance = d
                        self.point_distance_rs_ind_pairs[i] = p

    @ti.kernel
    def compute_point_euclidean_distance_rs_kernel(self, f: ti.i32):
        for i in range(self.n_target_pcd_points):
            p = self.point_distance_rs_ind_pairs[i]
            d = self.compute_euclidean_distance(self.particle_x[f, p] * 1000,
                                                self.target_pcd_points[i])
            self.avg_point_distance_rs[None] += d

    @ti.kernel
    def compute_point_euclidean_distance_sr_ind_pairs(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particle_mat[p] == self.matching_mat:
                smallest_distance: DTYPE_TI = 100
                for i in range(self.n_target_pcd_points):
                    d = self.compute_euclidean_distance(self.particle_x[f, p] * 1000,
                                                        self.target_pcd_points[i])
                    if d < smallest_distance:
                        smallest_distance = d
                        self.point_distance_sr_ind_pairs[p] = i

    @ti.kernel
    def compute_point_euclidean_distance_sr_kernel(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particle_mat[p] == self.matching_mat:
                i = self.point_distance_sr_ind_pairs[p]
                d = self.compute_euclidean_distance(self.particle_x[f, p] * 1000,
                                                    self.target_pcd_points[i])
                self.avg_point_distance_sr[None] += d

    """Particle euclidean distance loss"""
    @ti.kernel
    def compute_particle_euclidean_distance_sr_ind_pairs(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particle_mat[p] == self.matching_mat:
                smallest_distance: DTYPE_TI = 100
                for i in range(self.n_target_particles_from_mesh):
                    d = self.compute_euclidean_distance(self.particle_x[f, p] * 1000,
                                                        self.target_particles_from_mesh[i])
                    if d < smallest_distance:
                        smallest_distance = d
                        self.particle_distance_sr_ind_pairs[p] = i

    @ti.kernel
    def compute_particle_euclidean_distance_sr_kernel(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particle_mat[p] == self.matching_mat:
                i = self.particle_distance_sr_ind_pairs[p]
                d = self.compute_euclidean_distance(self.particle_x[f, p] * 1000,
                                                    self.target_particles_from_mesh[i])
                self.avg_particle_distance_sr[None] += d

    @ti.kernel
    def compute_particle_euclidean_distance_rs_ind_pairs(self, f: ti.i32):
        for i in range(self.n_target_particles_from_mesh):
            smallest_distance: DTYPE_TI = 100
            for p in range(self.n_particles):
                if self.particle_mat[p] == self.matching_mat:
                    d = self.compute_euclidean_distance(self.particle_x[f, p] * 1000,
                                                        self.target_particles_from_mesh[i])
                    if d < smallest_distance:
                        smallest_distance = d
                        self.particle_distance_rs_ind_pairs[i] = p

    @ti.kernel
    def compute_particle_euclidean_distance_rs_kernel(self, f: ti.i32):
        for i in range(self.n_target_particles_from_mesh):
            p = self.particle_distance_rs_ind_pairs[i]
            d = self.compute_euclidean_distance(self.particle_x[f, p] * 1000,
                                                self.target_particles_from_mesh[i])
            self.avg_particle_distance_rs[None] += d

    def compute_chamfer_loss(self, f):
        if self.exponential_distance:
            self.compute_point_exponential_distance_sr_ind_pairs(f)
            self.compute_point_exponential_distance_sr_kernel(f)
            self.compute_point_exponential_distance_rs_ind_pairs(f)
            self.compute_point_exponential_distance_rs_kernel(f)
            self.compute_particle_exponential_distance_sr_ind_pairs(f)
            self.compute_particle_exponential_distance_sr_kernel(f)
            self.compute_particle_exponential_distance_rs_ind_pairs(f)
            self.compute_particle_exponential_distance_rs_kernel(f)
        else:
            self.compute_point_euclidean_distance_sr_ind_pairs(f)
            self.compute_point_euclidean_distance_sr_kernel(f)
            self.compute_point_euclidean_distance_rs_ind_pairs(f)
            self.compute_point_euclidean_distance_rs_kernel(f)
            self.compute_particle_euclidean_distance_sr_ind_pairs(f)
            self.compute_particle_euclidean_distance_sr_kernel(f)
            self.compute_particle_euclidean_distance_rs_ind_pairs(f)
            self.compute_particle_euclidean_distance_rs_kernel(f)

    def compute_chamfer_loss_grad(self, f):
        if self.exponential_distance:
            self.compute_particle_exponential_distance_rs_kernel.grad(f)
            self.compute_particle_exponential_distance_sr_kernel.grad(f)
            self.compute_point_exponential_distance_rs_kernel.grad(f)
            self.compute_point_exponential_distance_sr_kernel.grad(f)
        else:
            self.compute_particle_euclidean_distance_rs_kernel.grad(f)
            self.compute_particle_euclidean_distance_sr_kernel.grad(f)
            self.compute_point_euclidean_distance_rs_kernel.grad(f)
            self.compute_point_euclidean_distance_sr_kernel.grad(f)

    """EMD loss"""
    @ti.kernel
    def compute_emd_point_euclidean_distance_matrix(self, f: ti.i32):
        for i in range(self.n_target_pcd_points):
            for j in range(self.n_particles):
                if self.particle_mat[j] == self.matching_mat:
                    self.emd_point_distance_matrix[i, j] = self.compute_euclidean_distance(self.particle_x[f, j] * 1000,
                                                                                           self.target_pcd_points[i])

    @ti.kernel
    def compute_emd_point_exponential_distance_matrix(self, f: ti.i32):
        for i in range(self.n_target_pcd_points):
            for j in range(self.n_particles):
                if self.particle_mat[j] == self.matching_mat:
                    self.emd_point_distance_matrix[i, j] = self.compute_exponential_euclidean_distance(self.particle_x[f, j] * 1000,
                                                                                                       self.target_pcd_points[i])

    def compute_emd_point_distance_bijection(self):
        mat = self.emd_point_distance_matrix.to_numpy()
        attempt = 0
        done = False
        while not done:
            attempt += 1
            try:
                ind1, ind2 = linear_sum_assignment(mat)
                indexes = np.stack((ind1, ind2), axis=-1).astype(np.int32)
                done = True
            except:
                print('Error: linear_sum_assignment failed')
                print(f'D mat shape: {mat.shape}')
                print(
                    f'D mat NaN: {np.isnan(mat).any()}, Inf: {np.isinf(mat).any()}, Max: {np.max(mat)}, Min: {np.min(mat)}')
                print(f'Target pcd shape: {self.target_pcd_points_np.shape}')
                print(
                    f'Target pcd NaN: {np.isnan(self.target_pcd_points_np).any()}, Inf: {np.isinf(self.target_pcd_points_np).any()}, '
                    f'Max: {np.max(self.target_pcd_points_np)}, Min: {np.min(self.target_pcd_points_np)}')
                if self.logger is not None:
                    self.logger.error('Error: linear_sum_assignment failed')
                    self.logger.error(f'D mat shape: {mat.shape}')
                    self.logger.error(f'D mat NaN: {np.isnan(mat).any()}, Inf: {np.isinf(mat).any()}, Max: {np.max(mat)}, Min: {np.min(mat)}')
                    self.logger.error(f'Target pcd shape: {self.target_pcd_points_np.shape}')
                    self.logger.error(f'Target pcd NaN: {np.isnan(self.target_pcd_points_np).any()}, Inf: {np.isinf(self.target_pcd_points_np).any()}, '
                                      f'Max: {np.max(self.target_pcd_points_np)}, Min: {np.min(self.target_pcd_points_np)}')
            if attempt > 5:
                print('Linear assignment failed for 5 times, exiting calculation.')
                if self.logger is not None:
                    self.logger.error('Linear assignment failed for 5 times, exiting calculation.')
                done = True
                self.pcd_linear_assignment_failed = True
        if done:
            self.emd_point_ind_pairs.from_numpy(indexes)

    @ti.kernel
    def compute_emd_point_euclidean_distance(self, f: ti.i32):
        for n in range(self.n_target_pcd_points):
            i = self.emd_point_ind_pairs[n][0]
            j = self.emd_point_ind_pairs[n][1]
            d = self.compute_euclidean_distance(self.target_pcd_points[i],
                                                self.particle_x[f, j] * 1000)
            self.emd_point_loss[None] += d

    @ti.kernel
    def compute_emd_point_exponential_distance(self, f: ti.i32):
        for n in range(self.n_target_pcd_points):
            i = self.emd_point_ind_pairs[n][0]
            j = self.emd_point_ind_pairs[n][1]
            d = self.compute_exponential_euclidean_distance(self.target_pcd_points[i],
                                                            self.particle_x[f, j] * 1000)
            self.emd_point_loss[None] += d

    @ti.kernel
    def compute_emd_particle_euclidean_distance_matrix(self, f: ti.i32):
        for i in range(self.n_target_particles_from_mesh):
            for j in range(self.n_particles):
                if self.particle_mat[j] == self.matching_mat:
                    self.emd_particle_distance_matrix[i, j] = self.compute_euclidean_distance(self.particle_x[f, j] * 1000,
                                                                                              self.target_particles_from_mesh[i])

    @ti.kernel
    def compute_emd_particle_exponential_distance_matrix(self, f: ti.i32):
        for i in range(self.n_target_particles_from_mesh):
            for j in range(self.n_particles):
                if self.particle_mat[j] == self.matching_mat:
                    self.emd_particle_distance_matrix[i, j] = self.compute_exponential_euclidean_distance(self.particle_x[f, j] * 1000,
                                                                                                          self.target_particles_from_mesh[i])

    def compute_emd_particle_distance_bijection(self):
        mat = self.emd_particle_distance_matrix.to_numpy()
        attempt = 0
        done = False
        while not done:
            attempt += 1
            try:
                ind1, ind2 = linear_sum_assignment(mat)
                indexes = np.stack((ind1, ind2), axis=-1).astype(np.int32)
                done = True
            except:
                print('Error: linear_sum_assignment failed')
                print(mat.shape)
                print(np.isnan(mat).any(), np.isinf(mat).any(), np.max(mat), np.min(mat))
                target_particle_np = self.target_particles_from_mesh.to_numpy()
                print(target_particle_np.shape)
                print(np.isnan(target_particle_np).any(), np.isinf(target_particle_np).any(),
                        np.max(target_particle_np), np.min(target_particle_np))
                if self.logger is not None:
                    self.logger.error('Error: linear_sum_assignment failed')
                    self.logger.error(f'D mat shape: {mat.shape}')
                    self.logger.error(
                        f'D mat NaN: {np.isnan(mat).any()}, Inf: {np.isinf(mat).any()}, Max: {np.max(mat)}, Min: {np.min(mat)}')
                    self.logger.error(f'Target particle shape: {target_particle_np.shape}')
                    self.logger.error(
                        f'Target particle NaN: {np.isnan(target_particle_np).any()}, Inf: {np.isinf(target_particle_np).any()}, '
                        f'Max: {np.max(target_particle_np)}, Min: {np.min(target_particle_np)}')
            if attempt > 5:
                print('Linear assignment failed for 5 times, exiting calculation.')
                if self.logger is not None:
                    self.logger.error('Linear assignment failed for 5 times, exiting calculation.')
                done = True
                self.particle_linear_assignment_failed = True

        if done:
            self.emd_particle_ind_pairs.from_numpy(indexes)

    @ti.kernel
    def compute_emd_particle_euclidean_distance(self, f: ti.i32):
        for n in range(self.n_target_particles_from_mesh):
            i = self.emd_particle_ind_pairs[n][0]
            j = self.emd_particle_ind_pairs[n][1]
            d = self.compute_euclidean_distance(self.target_particles_from_mesh[i],
                                                self.particle_x[f, j] * 1000)
            self.emd_particle_loss[None] += d

    @ti.kernel
    def compute_emd_particle_exponential_distance(self, f: ti.i32):
        for n in range(self.n_target_particles_from_mesh):
            i = self.emd_particle_ind_pairs[n][0]
            j = self.emd_particle_ind_pairs[n][1]
            d = self.compute_exponential_euclidean_distance(self.target_particles_from_mesh[i],
                                                            self.particle_x[f, j] * 1000)
            self.emd_particle_loss[None] += d

    def compute_emd_loss(self, f):
        self.emd_point_ind_pairs.fill(-1)
        self.emd_point_distance_matrix.fill(10000.0)
        if not self.exponential_distance:
            self.compute_emd_point_euclidean_distance_matrix(f)
        else:
            self.compute_emd_point_exponential_distance_matrix(f)
        self.pcd_linear_assignment_failed = False
        self.compute_emd_point_distance_bijection()
        if not self.pcd_linear_assignment_failed:
            if not self.exponential_distance:
                self.compute_emd_point_euclidean_distance(f)
            else:
                self.compute_emd_point_exponential_distance(f)  # the only operation that needs gradients

        self.emd_particle_ind_pairs.fill(-1)
        self.emd_particle_distance_matrix.fill(10000.0)
        if not self.exponential_distance:
            self.compute_emd_particle_euclidean_distance_matrix(f)
        else:
            self.compute_emd_particle_exponential_distance_matrix(f)
        self.particle_linear_assignment_failed = False
        self.compute_emd_particle_distance_bijection()
        if not self.particle_linear_assignment_failed:
            if not self.exponential_distance:
                self.compute_emd_particle_euclidean_distance(f)
            else:
                self.compute_emd_particle_exponential_distance(f)  # the only operation that needs gradients

    def compute_emd_loss_grad(self, f):
        if not self.exponential_distance:
            self.compute_emd_particle_euclidean_distance.grad(f)
            self.compute_emd_point_euclidean_distance.grad(f)
        else:
            self.compute_emd_particle_exponential_distance.grad(f)
            self.compute_emd_point_exponential_distance.grad(f)

    @ti.kernel
    def averaging_loss_kernel(self):
        self.avg_point_distance_sr[None] /= self.n_particles_matching_mat
        self.avg_point_distance_rs[None] /= self.n_target_pcd_points
        self.avg_particle_distance_sr[None] /= self.n_particles_matching_mat
        self.avg_particle_distance_rs[None] /= self.n_target_particles_from_mesh
        self.height_map_loss_pcd[None] /= (self.height_map_res ** 2)
        self.emd_point_loss[None] /= self.n_target_pcd_points
        self.emd_particle_loss[None] /= self.n_target_particles_from_mesh

    @ti.kernel
    def sum_up_loss_kernel(self):
        cf_pcd = self.avg_point_distance_rs[None] + self.avg_point_distance_sr[None]
        self.chamfer_loss_pcd[None] = cf_pcd
        cf_prd = self.avg_particle_distance_rs[None] + self.avg_particle_distance_sr[None]
        self.chamfer_loss_particle[None] = cf_prd

    @ti.kernel
    def get_final_loss_kernel(self):
        if ti.static(self.height_map_loss):
            self.total_loss[None] += self.height_map_loss_pcd[None]
        if ti.static(self.point_distance_rs_loss):
            self.total_loss[None] += self.avg_point_distance_rs[None]
        if ti.static(self.point_distance_sr_loss):
            self.total_loss[None] += self.avg_point_distance_sr[None]
        if ti.static(self.particle_distance_rs_loss):
            self.total_loss[None] += self.avg_particle_distance_rs[None]
        if ti.static(self.particle_distance_sr_loss):
            self.total_loss[None] += self.avg_particle_distance_sr[None]
        if ti.static(self.emd_point_distance_loss):
            self.total_loss[None] += self.emd_point_loss[None]
        if ti.static(self.emd_particle_distance_loss):
            self.total_loss[None] += self.emd_particle_loss[None]

    def validate_nan_inf_particles(self, f):
        particles = self.particle_x.to_numpy()[f]
        if np.any(np.isnan(particles)) or np.any(np.isinf(particles)):
            return True
        else:
            return False

    def get_final_loss(self):
        particle_has_naninf = self.validate_nan_inf_particles(self.sim.cur_substep_local)
        if not particle_has_naninf:
            self.compute_height_map_loss(self.sim.cur_substep_local)
            self.compute_chamfer_loss(self.sim.cur_substep_local)
            self.compute_emd_loss(self.sim.cur_substep_local)
            if self.averaging_loss:
                self.averaging_loss_kernel()
            self.sum_up_loss_kernel()
            self.get_final_loss_kernel()

        loss_info = {
            'particle_has_naninf': particle_has_naninf,
            'point_distance_sr': self.avg_point_distance_sr[None],
            'point_distance_rs': self.avg_point_distance_rs[None],
            'chamfer_loss_pcd': self.chamfer_loss_pcd[None],
            'particle_distance_sr': self.avg_particle_distance_sr[None],
            'particle_distance_rs': self.avg_particle_distance_rs[None],
            'chamfer_loss_particle': self.chamfer_loss_particle[None],
            'final_height_map': self.height_map.to_numpy(),
            'height_map_loss_pcd': self.height_map_loss_pcd[None],
            'emd_point_distance_loss': self.emd_point_loss[None],
            'emd_particle_distance_loss': self.emd_particle_loss[None],
            'total_loss': self.total_loss[None],
        }
        return loss_info

    def get_final_loss_grad(self):
        self.get_final_loss_kernel.grad()
        self.sum_up_loss_kernel.grad()
        if self.averaging_loss:
            self.averaging_loss_kernel.grad()
        self.compute_emd_loss_grad(self.sim.cur_substep_local)
        self.compute_chamfer_loss_grad(self.sim.cur_substep_local)
        self.compute_height_map_loss_grad(self.sim.cur_substep_local)

        # print(f'******Grads after get_final_loss_grad()')
        # print(f"Gradient of total loss: {self.total_loss.grad[None]}")
        # print(f"Gradient of chamfer_loss_pcd: {self.chamfer_loss_pcd.grad[None]}")
        # print(f"Gradient of chamfer_loss_particle: {self.chamfer_loss_particle.grad[None]}")
        # print(f"Gradient of emd_particle_loss: {self.emd_particle_loss.grad[None]}")
        # print(f"Gradient of emd_point_loss: {self.emd_point_loss.grad[None]}")
        # print(f"Gradient of avg_particle_distance_rs: {self.avg_particle_distance_rs.grad[None]}")
        # print(f"Gradient of avg_particle_distance_sr: {self.avg_particle_distance_sr.grad[None]}")
        # print(f"Gradient of avg_point_distance_sr: {self.avg_point_distance_sr.grad[None]}")
        # print(f"Gradient of avg_point_distance_rs: {self.avg_point_distance_rs.grad[None]}")
        # print(f"Gradient of height_map_loss_pcd: {self.height_map_loss_pcd.grad[None]}")
