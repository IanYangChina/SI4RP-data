import numpy as np
import taichi as ti
from doma.engine.configs.macros import DTYPE_NP, DTYPE_TI, EPS
from doma.engine.utils.misc import eval_str


@ti.data_oriented
class Boundary:
    def __init__(self, restitution=0.0, lock_dims=[]):
        self.restitution = restitution
        self.lock_dims = lock_dims

    @ti.func
    def impose_x_v(self, x, v, ground_friction=0):
        raise NotImplementedError

    @ti.func
    def impose_x(self, x):
        raise NotImplementedError

    @ti.func
    def is_out(self, x):
        raise NotImplementedError


@ti.data_oriented
class CylinderBoundary(Boundary):
    def __init__(self, y_range=(0.05, 0.95), xz_center=(0.5, 0.5), xz_radius=0.45, **kwargs):
        super(CylinderBoundary, self).__init__(**kwargs)

        y_range = np.array(eval_str(y_range), dtype=DTYPE_NP)
        xz_center = np.array(eval_str(xz_center), dtype=DTYPE_NP)
        self.y_lower = ti.Vector([0.0, y_range[0], 0.0], dt=DTYPE_TI)
        self.y_upper = ti.Vector([1.0, y_range[1], 1.0], dt=DTYPE_TI)
        self.xz_center = ti.Vector(xz_center, dt=DTYPE_TI)
        self.xz_radius = xz_radius

    @ti.func
    def impose_x_v(self, x, v):
        # y direction
        if x[1] > self.y_upper[1] and v[1] > 0.0:
            v[1] *= -self.restitution
        elif x[1] < self.y_lower[1] and v[1] < 0.0:
            v[1] *= -self.restitution

        x_new = ti.max(ti.min(x, self.y_upper), self.y_lower)

        # xz direction
        r_vector = ti.Vector([x[0], x[2]]) - self.xz_center
        r_vector_norm = r_vector.norm(EPS)
        if r_vector_norm > self.xz_radius:
            new_xz = r_vector / r_vector_norm * self.xz_radius + self.xz_center
            new_y = x_new[1]
            x_new = ti.Vector([new_xz[0], new_y, new_xz[1]])
            v[0] = 0.0
            v[2] = 0.0

        # enforce lock_dims
        for i in ti.static(self.lock_dims):
            v[i] = 0.0

        return x_new, v

    @ti.func
    def impose_x(self, x):
        # y direction
        x_new = ti.max(ti.min(x, self.y_upper), self.y_lower)

        # xz direction
        r_vector = ti.Vector([x[0], x[2]]) - self.xz_center
        r_vector_norm = r_vector.norm(EPS)
        if r_vector_norm > self.xz_radius:
            new_xz = r_vector / r_vector_norm * self.xz_radius + self.xz_center
            new_y = x_new[1]
            x_new = ti.Vector([new_xz[0], new_y, new_xz[1]])

        return x_new

    @ti.func
    def is_out(self, x):
        out = False

        # y direction
        if x[1] > self.y_upper[1] or x[1] < self.y_lower[1]:
            out = True

        # xz direction
        r_vector = ti.Vector([x[0], x[2]]) - self.xz_center
        r_vector_norm = r_vector.norm(EPS)
        if r_vector_norm > self.xz_radius:
            out = True
        return out


@ti.data_oriented
class CubeBoundary(Boundary):
    def __init__(self, lower=(0.05, 0.05, 0.05), upper=(0.95, 0.95, 0.95), **kwargs):
        super(CubeBoundary, self).__init__(**kwargs)
        self.upper_tuple = upper
        self.lower_tuple = lower
        upper = np.array(eval_str(upper), dtype=DTYPE_NP)
        lower = np.array(eval_str(lower), dtype=DTYPE_NP)
        assert (upper >= lower).all()

        self.upper_original = ti.Vector(upper, dt=DTYPE_TI)
        self.lower_original = ti.Vector(lower, dt=DTYPE_TI)
        self.upper = ti.Vector.field(3, dtype=DTYPE_TI, shape=())
        self.lower = ti.Vector.field(3, dtype=DTYPE_TI, shape=())
        self.upper.from_numpy(upper.astype(DTYPE_NP))
        self.lower.from_numpy(lower.astype(DTYPE_NP))

    @ti.func
    def impose_x_v(self, x, v, ground_friction):
        for i in ti.static(range(3)):
            if x[i] >= self.upper[None][i] and v[i] >= 0:
                v[i] *= -self.restitution
            elif x[i] <= self.lower[None][i] and v[i] <= 0:
                if ti.static(i != 2):
                    v[i] *= -self.restitution
                else:
                    # ground contact friction handling from:
                    # https://github.com/taichi-dev/difftaichi/blob/2d6e6cd3b2e5e034be57fba86749b75384fd6215/examples/diffmpm3d.py#L199
                    if ground_friction <= 2:
                        v_bound = ti.Vector.zero(dt=DTYPE_TI, n=3)
                        v_rel = v - v_bound
                        normal = ti.Vector.zero(dt=DTYPE_TI, n=3)
                        normal[2] = 1.0
                        v_rel_normal_direction = v_rel.dot(normal)
                        if v_rel_normal_direction < 0:
                            v_rel_tangent_portion = v_rel - v_rel_normal_direction * normal + 1e-10
                            v_rel_tangent_portion_norm = v_rel_tangent_portion.norm() + 1e-10
                            if v_rel_tangent_portion_norm + ground_friction * v_rel_normal_direction > 0:
                                v_rel_new = (1 + ground_friction * v_rel_normal_direction / v_rel_tangent_portion_norm) * v_rel_tangent_portion
                                v = v_rel_new + v_bound
                            else:
                                v = ti.Vector.zero(dt=DTYPE_TI, n=3)
                    else:
                        # ground_friction > 15, sticky boundary
                        v = ti.Vector.zero(dt=DTYPE_TI, n=3)

        x_new = ti.max(ti.min(x, self.upper[None]), self.lower[None])

        # enforce lock_dims
        for i in ti.static(self.lock_dims):
            v[i] = 0.0

        return x_new, v

    @ti.func
    def impose_x(self, x):
        x_new = ti.max(ti.min(x, self.upper[None]), self.lower[None])
        return x_new

    @ti.func
    def is_out(self, x):
        out = False

        if any(x > self.upper[None]) or any(x < self.lower[None]):
            out = True

        return out

    @ti.func
    def update_boundary(self, lower, upper):
        self.lower[None] = lower
        self.upper[None] = upper


def create_boundary(type='cube', **kwargs):
    if type == 'cylinder':
        return CylinderBoundary(**kwargs)
    if type == 'cube':
        return CubeBoundary(**kwargs)
    else:
        assert False
