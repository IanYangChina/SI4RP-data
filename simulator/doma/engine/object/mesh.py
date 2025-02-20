import os
import trimesh
import taichi as ti
import pickle as pkl
import numpy as np
from doma.engine.configs.macros import COLOR, DTYPE_NP, DTYPE_TI, FRICTION, EPS
from scipy.spatial.transform import Rotation
from doma.engine.utils.misc import eval_str
import doma.engine.utils.mesh_ops as mesh_ops
import doma.engine.utils.transform as transform
import doma.engine.utils.transform_ti as transform_ti


@ti.data_oriented
class Mesh:
    def __init__(
            self,
            file,
            material,
            file_vis=None,
            sdf_res=128,
            pos=(0.0, 0.0, 0.0),
            euler=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            # softness=0,
            has_dynamics=False
    ):
        self.pos = eval_str(pos)
        self.euler = eval_str(euler)
        self.scale = eval_str(scale)
        self.raw_file = file
        self.sdf_res = sdf_res
        self.raw_file_vis = file if file_vis is None else file_vis
        self.material = eval_str(material)
        self.has_dynamics = has_dynamics
        # self.softness = softness
        self.gl_renderer_id = None

        self.load_file()
        self.init_transform()

    def load_file(self):
        # mesh
        self.process_mesh()
        self.mesh = trimesh.load(self.processed_file_path)
        self.raw_vertices = np.array(self.mesh.vertices, dtype=np.float32)
        self.raw_vertex_normals_np = np.array(self.mesh.vertex_normals, dtype=np.float32)
        self.faces_np = np.array(self.mesh.faces, dtype=np.int32).flatten()

        self.n_vertices = len(self.raw_vertices)
        self.n_faces = len(self.faces_np)

        # load color
        vcolor_path = self.raw_file_path.replace('obj', 'vcolor')
        if os.path.exists(vcolor_path):
            self.colors_np = pkl.load(open(vcolor_path, 'rb')).astype(np.float32)
        else:
            # if vcolor file does not exist, get color based on material
            self.colors_np = np.tile([COLOR[self.material]], [self.n_vertices, 1]).astype(np.float32)

        if self.has_dynamics:
            # sdf
            self.friction = FRICTION[self.material]
            sdf_data = pkl.load(open(self.processed_sdf_path, 'rb'))
            self.sdf_voxels_np = sdf_data['voxels'].astype(DTYPE_NP)
            self.sdf_voxels_res = self.sdf_voxels_np.shape[0]
            self.T_mesh_to_voxels_np = sdf_data['T_mesh_to_voxels'].astype(DTYPE_NP)

    def process_mesh(self):
        self.raw_file_path = mesh_ops.get_raw_mesh_path(self.raw_file)
        self.raw_file_vis_path = mesh_ops.get_raw_mesh_path(self.raw_file_vis)
        self.processed_file_path = mesh_ops.get_processed_mesh_path(self.raw_file, self.raw_file_vis)
        if self.has_dynamics:
            self.processed_sdf_path = mesh_ops.get_processed_sdf_path(self.raw_file, self.sdf_res)

        # clean up mesh
        if not os.path.exists(self.processed_file_path):
            print(f'===> Processing mesh(es) {self.raw_file_path} and vis {self.raw_file_vis_path}.')
            raw_mesh = mesh_ops.load_mesh(self.raw_file_path)

            # process and save
            processed_mesh = mesh_ops.cleanup_mesh(raw_mesh)
            processed_mesh.export(self.processed_file_path)
            print(f'===> Processed mesh saved as {self.processed_file_path}.')

        # generate sdf
        if self.has_dynamics and not os.path.exists(self.processed_sdf_path):
            print(f'===> Computing sdf for {self.raw_file_path}. This might take minutes...')
            raw_mesh = mesh_ops.load_mesh(self.raw_file_path)
            processed_mesh_sdf = mesh_ops.cleanup_mesh(raw_mesh)
            sdf_data = mesh_ops.compute_sdf_data(processed_mesh_sdf, self.sdf_res)

            pkl.dump(sdf_data, open(self.processed_sdf_path, 'wb'))
            print(f'===> sdf saved as {self.processed_sdf_path}.')

    def init_transform(self):
        scale = np.array(self.scale, dtype=DTYPE_NP)
        pos = np.array(self.pos, dtype=DTYPE_NP)
        quat = transform.xyzw_to_wxyz(
            Rotation.from_euler('zyx', self.euler[::-1], degrees=True).as_quat().astype(DTYPE_NP))

        # apply initial transforms (scale then quat then pos)
        T_init = transform.trans_quat_to_T(pos, quat) @ transform.scale_to_T(scale)
        self.init_vertices_np = transform.transform_by_T_np(self.raw_vertices, T_init).astype(np.float32)
        self.init_vertices_np_flattened = self.init_vertices_np.flatten()

        R_init = transform.trans_quat_to_T(None, quat)
        self.init_vertex_normals_np = transform.transform_by_T_np(self.raw_vertex_normals_np, R_init).astype(
            np.float32)
        self.init_vertex_normals_np_flattened = self.init_vertex_normals_np.flatten()

        # init ti fields
        self.init_vertices = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_vertices))
        self.init_vertex_normals = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_vertices))
        self.faces = ti.field(dtype=ti.i32, shape=(self.n_faces))
        self.colors = ti.Vector.field(self.colors_np.shape[1], dtype=ti.f32, shape=(self.n_vertices))

        self.init_vertices.from_numpy(self.init_vertices_np)
        self.init_vertex_normals.from_numpy(self.init_vertex_normals_np)
        self.faces.from_numpy(self.faces_np)
        self.colors.from_numpy(self.colors_np)

        if self.has_dynamics:
            self.T_mesh_to_voxels_np = self.T_mesh_to_voxels_np @ np.linalg.inv(T_init)
            self.sdf_voxels = ti.field(dtype=DTYPE_TI, shape=self.sdf_voxels_np.shape)
            self.T_mesh_to_voxels = ti.Matrix.field(4, 4, dtype=DTYPE_TI, shape=())

            self.sdf_voxels.from_numpy(self.sdf_voxels_np)
            self.T_mesh_to_voxels.from_numpy(self.T_mesh_to_voxels_np)

        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_vertices))
        self.vertex_normals = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_vertices))

    def update_color(self, color):
        self.colors.from_numpy(np.tile([color], [self.n_vertices, 1]).astype(np.float32))


@ti.data_oriented
class Static(Mesh):
    # Static mesh-based object
    def __init__(self, **kwargs):
        super(Static, self).__init__(**kwargs)

    def init_transform(self):
        super(Static, self).init_transform()
        self.vertices.copy_from(self.init_vertices)
        self.vertex_normals.copy_from(self.init_vertex_normals)

    @ti.func
    def sdf(self, pos_world):
        # sdf value from world coordinate
        pos_mesh = pos_world
        pos_voxels = transform_ti.transform_by_T_ti(pos_mesh, self.T_mesh_to_voxels[None], DTYPE_TI)

        return self.sdf_(pos_voxels)

    @ti.func
    def sdf_(self, pos_voxels):
        # sdf value from voxels coordinate
        base = ti.floor(pos_voxels, ti.i32)
        signed_dist = ti.cast(0.0, DTYPE_TI)
        if (base >= self.sdf_voxels_res - 1).any() or (base < 0).any():
            signed_dist = 1.0
        else:
            signed_dist = 0.0
            for offset in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
                voxel_pos = base + offset
                w_xyz = 1 - ti.abs(pos_voxels - voxel_pos)
                w = w_xyz[0] * w_xyz[1] * w_xyz[2]
                signed_dist += w * self.sdf_voxels[voxel_pos]

        return signed_dist

    @ti.func
    def normal(self, pos_world):
        # compute normal with finite difference
        pos_mesh = pos_world
        pos_voxels = transform_ti.transform_by_T_ti(pos_mesh, self.T_mesh_to_voxels[None], DTYPE_TI)
        normal_vec_voxels = self.normal_(pos_voxels)

        R_voxels_to_mesh = self.T_mesh_to_voxels[None][:3, :3].inverse()
        normal_vec_mesh = R_voxels_to_mesh @ normal_vec_voxels

        normal_vec_world = normal_vec_mesh
        normal_vec_world = transform_ti.normalize(normal_vec_world)

        return normal_vec_world

    @ti.func
    def normal_(self, pos_voxels):
        # since we are in voxels frame, delta can be a relatively big value
        delta = ti.cast(1e-2, DTYPE_TI)
        normal_vec = ti.Vector([0, 0, 0], dt=DTYPE_TI)

        for i in ti.static(range(3)):
            inc = pos_voxels
            dec = pos_voxels
            inc[i] += delta
            dec[i] -= delta
            normal_vec[i] = (self.sdf_(inc) - self.sdf_(dec)) / (2 * delta)

        normal_vec = transform_ti.normalize(normal_vec)
        return normal_vec

    @ti.func
    def collide(self, pos_world, mat_v, friction=1.0):
        if ti.static(self.has_dynamics):
            signed_dist = self.sdf(pos_world)
            if signed_dist <= 0:
                collider_v = ti.Vector.zero(dt=DTYPE_TI, n=3)

                if friction > 2.0:
                    mat_v = collider_v
                else:
                    # v w.r.t collider
                    v_rel = mat_v - 0.0
                    normal = self.normal(pos_world)
                    v_rel_normal_direction = v_rel.dot(normal)
                    if v_rel_normal_direction < 0:
                        # collision happens
                        v_rel_tangent_portion = v_rel - v_rel_normal_direction * normal + 1e-10
                        v_rel_tangent_portion_norm = v_rel_tangent_portion.norm() + 1e-10
                        v_rel_new = ti.Vector.zero(dt=DTYPE_TI, n=3)
                        if v_rel_tangent_portion_norm + friction * v_rel_normal_direction > 0:
                            v_rel_new = (
                                                    1 + friction * v_rel_normal_direction / v_rel_tangent_portion_norm) * v_rel_tangent_portion

                        mat_v = v_rel_new

        return mat_v

    @ti.func
    def is_collide(self, pos_world):
        flag = 0
        if ti.static(self.has_dynamics):
            signed_dist = self.sdf(pos_world)
            if signed_dist <= 0:
                flag = 1

        return flag

    @ti.func
    def impose_x(self, pos_world):
        delta_pos = ti.Vector.zero(dt=DTYPE_TI, n=3)
        if ti.static(self.has_dynamics):
            signed_dist = self.sdf(pos_world)
            if signed_dist <= 0:
                normal = self.normal(pos_world)
                delta_pos = -signed_dist * normal
        return delta_pos


class Statics:
    # Static objects, where each one is a Mesh
    def __init__(self):
        self.statics = []

    def add_static(self, **kwargs):
        self.statics.append(Static(**kwargs))

    def __getitem__(self, index):
        return self.statics[index]

    def __len__(self):
        return len(self.statics)


@ti.data_oriented
class Dynamic(Mesh):
    # Dynamic mesh-based object
    def __init__(self, container, **kwargs):
        self.container = container
        self.vertice_max = ti.Vector.field(3, DTYPE_TI, shape=())
        self.vertice_min = ti.Vector.field(3, DTYPE_TI, shape=())
        super(Dynamic, self).__init__(**kwargs)

    def init_transform(self):
        super(Dynamic, self).init_transform()

    @ti.kernel
    def update_vertices(self, f: ti.i32):
        for i in self.vertices:
            self.vertices[i] = ti.cast(
                transform_ti.transform_by_trans_quat_ti(self.init_vertices[i], self.container.pos[f],
                                                        self.container.quat[f]), self.vertices.dtype)
            self.vertex_normals[i] = ti.cast(
                transform_ti.transform_by_quat_ti(self.init_vertex_normals[i], self.container.quat[f]),
                self.vertices.dtype)
            ti.atomic_max(self.vertice_max[None], self.vertices[i])
            ti.atomic_min(self.vertice_min[None], self.vertices[i])

    @ti.func
    def sdf(self, f, pos_world):
        # sdf value from world coordinate
        # into mesh frame
        pos_mesh = transform_ti.inv_transform_by_trans_quat_ti(pos_world, self.container.pos[f], self.container.quat[f])
        # into voxel (sdf) frame
        pos_voxels = transform_ti.transform_by_T_ti(pos_mesh, self.T_mesh_to_voxels[None], DTYPE_TI)

        return self.sdf_(pos_voxels)

    @ti.func
    def sdf_toi(self, f, pos_world_toi, toi, dt):
        # sdf value from world coordinate
        # into mesh frame
        container_pos_v = (self.container.delta_pos[f]) / dt
        container_pos_toi = self.container.pos[f] + toi * container_pos_v
        container_euler_v = (self.container.delta_euler[f]) / dt
        container_delta_euler_toi = toi * container_euler_v
        container_quat_toi = transform_ti.qmul(self.container.quat[f],
                                               transform_ti.w2quat(container_delta_euler_toi, DTYPE_TI))
        pos_mesh_toi = transform_ti.inv_transform_by_trans_quat_ti(pos_world_toi, container_pos_toi, container_quat_toi)
        # into voxel (sdf) frame
        pos_voxels_toi = transform_ti.transform_by_T_ti(pos_mesh_toi, self.T_mesh_to_voxels[None], DTYPE_TI)

        return self.sdf_(pos_voxels_toi)

    @ti.func
    def sdf_(self, pos_voxels):
        # sdf value from voxels coordinate
        base = ti.floor(pos_voxels, ti.i32)
        signed_dist = ti.cast(0.0, DTYPE_TI)
        if (base >= self.sdf_voxels_res - 1).any() or (base < 0).any():
            signed_dist = 1.0
        else:
            signed_dist = 0.0
            for offset in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
                voxel_pos = base + offset
                sdf_voxel = self.sdf_voxels[voxel_pos]
                w_xyz = 1 - ti.abs(pos_voxels - voxel_pos)
                w = w_xyz[0] * w_xyz[1] * w_xyz[2]
                signed_dist += w * sdf_voxel

        return signed_dist

    @ti.func
    def normal(self, f, pos_world):
        # compute normal with finite difference
        pos_mesh = transform_ti.inv_transform_by_trans_quat_ti(pos_world, self.container.pos[f], self.container.quat[f])
        pos_voxels = transform_ti.transform_by_T_ti(pos_mesh, self.T_mesh_to_voxels[None], DTYPE_TI)
        normal_vec_voxels = self.normal_(pos_voxels)

        R_voxels_to_mesh = self.T_mesh_to_voxels[None][:3, :3].inverse()
        normal_vec_mesh = R_voxels_to_mesh @ normal_vec_voxels

        normal_vec_world = transform_ti.transform_by_quat_ti(normal_vec_mesh, self.container.quat[f])
        normal_vec_world = transform_ti.normalize(normal_vec_world)

        return normal_vec_world

    @ti.func
    def normal_toi(self, f, pos_world_toi, toi, dt):
        container_pos_v = (self.container.delta_pos[f]) / dt
        container_pos_toi = self.container.pos[f] + toi * container_pos_v
        container_euler_v = (self.container.delta_euler[f]) / dt
        container_delta_euler_toi = toi * container_euler_v
        container_quat_toi = transform_ti.qmul(self.container.quat[f],
                                               transform_ti.w2quat(container_delta_euler_toi, DTYPE_TI))
        # compute normal with finite difference
        pos_mesh = transform_ti.inv_transform_by_trans_quat_ti(pos_world_toi, container_pos_toi, container_quat_toi)
        pos_voxels = transform_ti.transform_by_T_ti(pos_mesh, self.T_mesh_to_voxels[None], DTYPE_TI)
        normal_vec_voxels = self.normal_(pos_voxels)

        R_voxels_to_mesh = self.T_mesh_to_voxels[None][:3, :3].inverse()
        normal_vec_mesh = R_voxels_to_mesh @ normal_vec_voxels

        normal_vec_world = transform_ti.transform_by_quat_ti(normal_vec_mesh, self.container.quat[f])
        normal_vec_world = transform_ti.normalize(normal_vec_world)

        return normal_vec_world

    @ti.func
    def normal_(self, pos_voxels):
        # since we are in voxels frame, delta can be a relatively big value
        delta = ti.cast(1e-2, DTYPE_TI)
        normal_vec = ti.Vector([0, 0, 0], dt=DTYPE_TI)

        for i in ti.static(range(3)):
            inc = pos_voxels
            dec = pos_voxels
            inc[i] += delta
            dec[i] -= delta
            normal_vec[i] = (self.sdf_(inc) - self.sdf_(dec)) / (2 * delta)

        normal_vec = transform_ti.normalize(normal_vec)

        return normal_vec

    @ti.func
    def collider_v(self, f, pos_world, dt):
        pos_mesh = transform_ti.inv_transform_by_trans_quat_ti(pos_world, self.container.pos[f], self.container.quat[f])
        pos_world_new = transform_ti.transform_by_trans_quat_ti(pos_mesh, self.container.pos[f + 1], self.container.quat[f + 1])
        collider_v = (pos_world_new - pos_world) / dt
        return collider_v

    @ti.func
    def collider_v_toi(self, f, pos_world, toi, dt):
        container_pos_v = (self.container.delta_pos[f]) / dt
        container_pos_toi = self.container.pos[f] + toi * container_pos_v
        container_euler_v = (self.container.delta_euler[f]) / dt
        container_delta_euler_toi = toi * container_euler_v
        container_quat_toi = transform_ti.qmul(self.container.quat[f],
                                               transform_ti.w2quat(container_delta_euler_toi, DTYPE_TI))

        pos_mesh = transform_ti.inv_transform_by_trans_quat_ti(pos_world, self.container.pos[f], self.container.quat[f])
        pos_world_new = transform_ti.transform_by_trans_quat_ti(pos_mesh, container_pos_toi, container_quat_toi)
        collider_v_toi = (pos_world_new - pos_world) / toi
        return collider_v_toi

    @ti.func
    def collide(self, f, pos_world, mat_v, dt, friction):
        # one of the codes here breaks the autodiff system without advanced_optimization=True
        if ti.static(self.has_dynamics):
            signed_dist = self.sdf(f + 1, pos_world)
            if signed_dist <= 0:
                collider_v = self.collider_v(f, pos_world, dt)

                if friction > 2.0:
                    # sticky surface
                    mat_v = collider_v
                else:
                    # v w.r.t collider
                    v_rel = mat_v - collider_v
                    normal = self.normal(f, pos_world)

                    v_rel_normal_direction = v_rel.dot(normal)
                    if v_rel_normal_direction < 0:
                        # collision happens
                        v_rel_tangent_portion = v_rel - v_rel_normal_direction * normal + 1e-10
                        v_rel_tangent_portion_norm = v_rel_tangent_portion.norm() + 1e-10
                        v_rel_new = ti.Vector.zero(dt=DTYPE_TI, n=3)
                        if v_rel_tangent_portion_norm + friction * v_rel_normal_direction > 0:
                            v_rel_new = (1 + friction * v_rel_normal_direction / v_rel_tangent_portion_norm) * v_rel_tangent_portion

                        mat_v = collider_v + v_rel_new

        return mat_v

    @ti.func
    def collide_soft(self, f, pos_world, mat_v, dt, friction):
        # one of the codes here breaks the autodiff system without advanced_optimization=True
        if ti.static(self.has_dynamics):
            signed_dist = self.sdf(f, pos_world)
            influence = ti.min(ti.exp(-signed_dist * 300.0), 1)
            if signed_dist <= 0 or influence > 0.1:
                collider_v = self.collider_v(f, pos_world, dt)

                if friction > 2.0:
                    # sticky surface
                    mat_v = collider_v
                else:
                    # v w.r.t collider
                    v_rel = mat_v - collider_v
                    normal = self.normal(f, pos_world)

                    v_rel_normal_direction = v_rel.dot(normal)
                    if v_rel_normal_direction < 0:
                        # collision happens
                        v_rel_tangent_portion = v_rel - v_rel_normal_direction * normal + 1e-10
                        v_rel_tangent_portion_norm = v_rel_tangent_portion.norm() + 1e-10
                        v_rel_new = ti.Vector.zero(dt=DTYPE_TI, n=3)
                        if v_rel_tangent_portion_norm + friction * v_rel_normal_direction > 0:
                            v_rel_new = (1 + friction * v_rel_normal_direction / v_rel_tangent_portion_norm) * v_rel_tangent_portion

                        mat_v = collider_v + v_rel_new * influence + v_rel * (1-influence)

        return mat_v

    @ti.func
    def collide_toi(self, f, pos_world_new, mat_v, dt, friction):
        # one of the codes here breaks the autodiff system without advanced_optimization=True
        delta_x = ti.Vector.zero(dt=DTYPE_TI, n=3)
        toi = 0.0
        mat_v_toi = mat_v
        if ti.static(self.has_dynamics):
            # impose x for the first time step
            pos_world_old = pos_world_new - mat_v * dt
            signed_dist_f = self.sdf(f, pos_world_old)
            if signed_dist_f < 0:
                normal = self.normal(f, pos_world_old)
                delta_x = -signed_dist_f * normal
            pos_world_old = pos_world_old + delta_x

            # collision detection at f + 1
            signed_dist_toi = signed_dist_dt = self.sdf(f + 1, pos_world_new)
            n_divide = 0
            if signed_dist_dt <= 0:
                low = 0.0
                high = dt
                for n in range(15):
                    toi = (low + high) * 0.5 + low
                    pos_world_toi = pos_world_old + mat_v * toi
                    signed_dist_toi = self.sdf_toi(f, pos_world_toi, toi, dt)
                    if signed_dist_toi < -1e-7:
                        n_divide += 1
                        high = toi
                    if signed_dist_toi > 1e-7:
                        n_divide += 1
                        low = toi

                pos_world_toi = pos_world_old + mat_v * toi
                collider_v_toi = self.collider_v_toi(f, pos_world_toi, toi, dt)

                if friction > 2.0:
                    # sticky surface
                    mat_v_toi = collider_v_toi
                else:
                    # v w.r.t collider
                    v_rel_toi = mat_v - collider_v_toi
                    normal_toi = self.normal_toi(f, pos_world_toi, toi, dt)

                    v_rel_normal_direction = v_rel_toi.dot(normal_toi)
                    if v_rel_normal_direction < 0:
                        # collision happens
                        v_rel_tangent_portion = v_rel_toi - v_rel_normal_direction * normal_toi + 1e-10
                        v_rel_tangent_portion_norm = v_rel_tangent_portion.norm() + 1e-10
                        v_rel_new = ti.Vector.zero(dt=DTYPE_TI, n=3)
                        if v_rel_tangent_portion_norm + friction * v_rel_normal_direction > 0:
                            v_rel_new = (1 + friction * v_rel_normal_direction / v_rel_tangent_portion_norm) * v_rel_tangent_portion

                        mat_v_toi = collider_v_toi + v_rel_new

                # print(n_divide, delta_x, toi, signed_dist_toi, signed_dist_dt, mat_v_toi)

        return delta_x, mat_v_toi, toi

    @ti.func
    def is_collide(self, f, pos_world):
        flag = 0
        if ti.static(self.has_dynamics):
            signed_dist = self.sdf(f, pos_world)
            if signed_dist <= 0:
                flag = 1

        return flag
