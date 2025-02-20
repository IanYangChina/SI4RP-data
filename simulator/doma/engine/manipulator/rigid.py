import taichi as ti
from doma.engine.manipulator.effector import Effector
from doma.engine.object.mesh import Dynamic
from doma.engine.configs.macros import DTYPE_TI


@ti.data_oriented
class Rigid(Effector):
    # Rigid end-effector. Can be a stirrer, ice-cream cone, laddle, whatever...
    def __init__(self, **kwargs):
        super(Rigid, self).__init__(**kwargs)
        self.mesh = None

        # magic, don't touch. (I finally got a chance to write something like this lol, thanks to stupid taichi)
        self.magic = ti.field(dtype=DTYPE_TI, shape=())
        self.magic.fill(0)

    def setup_mesh(self, **kwargs):
        self.mesh = Dynamic(
            container=self,
            has_dynamics=True,
            **kwargs
        )
        self.has_mesh = True

    def move(self, f):
        self.update_mesh_pose(f)
        self.update_boundary_from_mesh(f)
        # print(self.boundary.lower[None], self.boundary.upper[None])
        self.move_kernel(f)  # this updates the pose at f + 1

    def move_grad(self, f):
        self.move_kernel.grad(f)
        self.update_boundary_from_mesh.grad(f)

    def update_mesh_pose(self, f):
        # For visualization only. No need to compute grad.
        self.mesh.vertice_max.fill(-100)
        self.mesh.vertice_min.fill(100)
        self.mesh.update_vertices(f)

    @ti.func
    def collide(self, f, pos_world, mat_v, dt, friction):
        return self.mesh.collide(f, pos_world, mat_v, dt, friction)

    @ti.func
    def collide_soft(self, f, pos_world, mat_v, dt, friction):
        return self.mesh.collide_soft(f, pos_world, mat_v, dt, friction)

    @ti.func
    def collide_toi(self, f, pos_world, mat_v, dt, friction):
        return self.mesh.collide_toi(f, pos_world, mat_v, dt, friction)

    @ti.kernel
    def update_boundary_from_mesh(self, f: ti.i32):
        # rotation-aware action boundary
        if ti.static(self.has_mesh):
            b_min = self.boundary.lower_original + (self.pos[f] - self.mesh.vertice_min[None])
            b_max = self.boundary.upper_original - (self.mesh.vertice_max[None] - self.pos[f])
            self.boundary.update_boundary(b_min, b_max)

    def set_state(self, f, state):
        super(Rigid, self).set_state(f, state)
        self.update_mesh_pose(f)
        self.update_boundary_from_mesh(f)
