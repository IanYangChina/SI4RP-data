import torch
import taichi as ti
import numpy as np
from doma.engine.configs.macros import *
from doma.engine.utils.transform import xyzw_to_wxyz
from doma.engine.utils.transform_ti import qmul, w2quat
from doma.engine.utils.misc import *
from doma.engine.object.boundaries import create_boundary
from scipy.spatial.transform import Rotation


@ti.data_oriented
class Effector:
    # Effector base class
    state_dim = 7

    def __init__(
            self,
            max_substeps_local,
            max_substeps_global,
            max_action_steps_global,
            ckpt_dest,
            dt,
            ctrl_type='position',
            has_mesh=False,
            dim=3,
            action_dim=3,
            init_pos=(0.5, 0.5, 0.5),
            init_euler=(0.0, 0.0, 0.0),
    ):
        self.has_mesh = has_mesh
        self.dim = dim
        self.max_substeps_local = max_substeps_local  # this is f
        self.max_substeps_global = max_substeps_global  # this is f
        self.max_action_steps_global = max_action_steps_global  # this is s, not f
        self.ckpt_dest = ckpt_dest
        self.dt = dt
        self.ctrl_type = ctrl_type
        assert self.ctrl_type in ['velocity', 'position']

        self.pos = ti.Vector.field(3, DTYPE_TI, needs_grad=True)  # positon of the effector
        self.quat = ti.Vector.field(4, DTYPE_TI, needs_grad=True)  # quaternion for storing rotation

        self.delta_pos = ti.Vector.field(3, DTYPE_TI, needs_grad=True)  # velocity
        self.delta_euler = ti.Vector.field(3, DTYPE_TI, needs_grad=True)  # angular velocity

        ti.root.dense(ti.i, (self.max_substeps_local + 1,)).place(self.pos, self.pos.grad, self.quat, self.quat.grad,
                                                                  self.delta_pos, self.delta_pos.grad, self.delta_euler,
                                                                  self.delta_euler.grad)

        self.action_dim = action_dim
        self.init_pos = np.array(eval_str(init_pos))
        self.init_rot = xyzw_to_wxyz(Rotation.from_euler('zyx', eval_str(init_euler)[::-1], degrees=True).as_quat())

        if self.action_dim > 0:
            # each action in the action buffer passes (self.dt * n_substeps) seconds
            self.action_buffer = ti.Vector.field(self.action_dim, DTYPE_TI, needs_grad=True, shape=(max_action_steps_global,))
            self.action_buffer_p = ti.Vector.field(self.action_dim, DTYPE_TI, needs_grad=True, shape=())

        self.init_ckpt()

    def setup_boundary(self, **kwargs):
        self.boundary = create_boundary(**kwargs)

    def init_ckpt(self):
        if self.ckpt_dest == 'disk':
            self.pos_np = np.zeros((3), dtype=DTYPE_NP)
            self.quat_np = np.zeros((4), dtype=DTYPE_NP)
            self.delta_pos_np = np.zeros((3), dtype=DTYPE_NP)
            self.delta_euler_np = np.zeros((3), dtype=DTYPE_NP)
        elif self.ckpt_dest in ['cpu', 'gpu']:
            self.ckpt_ram = dict()

    def reset_grad(self):
        self.pos.grad.fill(0)
        self.quat.grad.fill(0)
        self.delta_pos.grad.fill(0)
        self.delta_euler.grad.fill(0)
        self.action_buffer.grad.fill(0)
        self.action_buffer_p.grad.fill(0)

    @ti.kernel
    def get_ckpt_kernel(self, pos_np: ti.types.ndarray(), quat_np: ti.types.ndarray(), v_np: ti.types.ndarray(),
                        w_np: ti.types.ndarray()):
        for i in ti.static(range(3)):
            pos_np[i] = self.pos[0][i]
            v_np[i] = self.delta_pos[0][i]
            w_np[i] = self.delta_euler[0][i]

        for i in ti.static(range(4)):
            quat_np[i] = self.quat[0][i]

    @ti.kernel
    def set_ckpt_kernel(self, pos_np: ti.types.ndarray(), quat_np: ti.types.ndarray(), v_np: ti.types.ndarray(),
                        w_np: ti.types.ndarray()):
        for i in ti.static(range(3)):
            self.pos[0][i] = pos_np[i]
            self.delta_pos[0][i] = v_np[i]
            self.delta_euler[0][i] = w_np[i]

        for i in ti.static(range(4)):
            self.quat[0][i] = quat_np[i]

    def get_ckpt(self, ckpt_name=None):
        if self.ckpt_dest == 'disk':
            ckpt = {
                'pos': self.pos_np,
                'quat': self.quat_np,
                'delta_pos': self.delta_pos_np,
                'delta_euler': self.delta_euler_np,
            }
            self.get_ckpt_kernel(self.pos_np, self.quat_np, self.delta_pos_np, self.delta_euler_np)
            return ckpt

        elif self.ckpt_dest in ['cpu', 'gpu']:
            if not ckpt_name in self.ckpt_ram:
                if self.ckpt_dest == 'cpu':
                    device = 'cpu'
                elif self.ckpt_dest == 'gpu':
                    device = 'cuda'
                self.ckpt_ram[ckpt_name] = {
                    'pos': torch.zeros((3), dtype=DTYPE_TC, device=device),
                    'quat': torch.zeros((4), dtype=DTYPE_TC, device=device),
                    'delta_pos': torch.zeros((3), dtype=DTYPE_TC, device=device),
                    'delta_euler': torch.zeros((3), dtype=DTYPE_TC, device=device),
                }
            self.get_ckpt_kernel(
                self.ckpt_ram[ckpt_name]['pos'],
                self.ckpt_ram[ckpt_name]['quat'],
                self.ckpt_ram[ckpt_name]['delta_pos'],
                self.ckpt_ram[ckpt_name]['delta_euler'],
            )

    def set_ckpt(self, ckpt=None, ckpt_name=None):
        if self.ckpt_dest == 'disk':
            assert ckpt is not None

        elif self.ckpt_dest in ['cpu', 'gpu']:
            ckpt = self.ckpt_ram[ckpt_name]

        self.set_ckpt_kernel(ckpt['pos'], ckpt['quat'], ckpt['delta_pos'], ckpt['delta_euler'])

    def clear_ckpt(self):
        if self.ckpt_dest == 'disk':
            pass
        else:
            self.ckpt_ram = dict()

    def move(self, f):
        self.move_kernel(f)

    def move_grad(self, f):
        self.move_kernel.grad(f)

    @ti.kernel
    def move_kernel(self, f: ti.i32):
        self.pos[f + 1] = self.boundary.impose_x(self.pos[f] + self.delta_pos[f])
        # rotate in world coordinates about itself.
        self.quat[f + 1] = qmul(self.quat[f], w2quat(self.delta_euler[f], DTYPE_TI))

    # state set and copy ...
    @ti.func
    def copy_frame(self, source, target):
        self.pos[target] = self.pos[source]
        self.quat[target] = self.quat[source]
        self.delta_pos[target] = self.delta_pos[source]
        self.delta_euler[target] = self.delta_euler[source]

    @ti.func
    def copy_grad(self, source, target):
        self.pos.grad[target] = self.pos.grad[source]
        self.quat.grad[target] = self.quat.grad[source]
        self.delta_pos.grad[target] = self.delta_pos.grad[source]
        self.delta_euler.grad[target] = self.delta_euler.grad[source]

    @ti.func
    def reset_grad_till_frame(self, f):
        for i in range(f):
            self.pos.grad[i].fill(0)
            self.quat.grad[i].fill(0)
            self.delta_pos.grad[i].fill(0)
            self.delta_euler.grad[i].fill(0)

    @ti.kernel
    def get_state_kernel(self, f: ti.i32, controller: ti.types.ndarray()):
        for j in ti.static(range(3)):
            controller[j] = self.pos[f][j]
        for j in ti.static(range(4)):
            controller[j + self.dim] = self.quat[f][j]

    @ti.kernel
    def set_state_kernel(self, f: ti.i32, controller: ti.types.ndarray()):
        for j in ti.static(range(3)):
            self.pos[f][j] = controller[j]
        for j in ti.static(range(4)):
            self.quat[f][j] = controller[j + self.dim]

    def get_state(self, f):
        out = np.zeros((7), dtype=DTYPE_NP)
        self.get_state_kernel(f, out)
        return out

    def set_state(self, f, state):
        ss = self.get_state(f)
        ss[:len(state)] = state
        self.set_state_kernel(f, ss)

    @property
    def init_state(self):
        return np.append(self.init_pos, self.init_rot)

    def build(self):
        self.set_state(0, self.init_state)

    @ti.kernel
    def apply_action_p_kernel(self):
        self.pos[0] = self.boundary.impose_x(self.action_buffer_p[None][:3])
        # todo: rotation not quite right
        self.quat[0] = w2quat(self.action_buffer_p[None][3:], DTYPE_TI)

    def apply_action_p(self, action_p):
        action_p = action_p.astype(DTYPE_NP)
        self.set_action_p_kernel(action_p)
        self.apply_action_p_kernel()

    def apply_action_p_grad(self, action_p):
        self.apply_action_p_kernel.grad()

    @ti.kernel
    def set_action_p_kernel(self, action_p: ti.types.ndarray()):
        for j in ti.static(range(self.action_dim)):
            self.action_buffer_p[None][j] = action_p[j]

    @ti.kernel
    def get_action_v_grad_kernel(self, s: ti.i32, n: ti.i32, grad: ti.types.ndarray()):
        for i in range(0, n):
            for j in ti.static(range(self.action_dim)):
                grad[i, j] = self.action_buffer.grad[s + i][j]

    @ti.kernel
    def get_action_p_grad_kernel(self, n: ti.i32, grad: ti.types.ndarray()):
        for j in ti.static(range(self.action_dim)):
            grad[n, j] = self.action_buffer_p.grad[None][j]

    @ti.kernel
    def set_delta_pos_by_velocity(self, s: ti.i32, s_global: ti.i32, n_substeps: ti.i32):
        # each global step passes self.dt_global seconds, divided into n_substeps
        # pos change = action * dt
        for j in range(s * n_substeps, (s + 1) * n_substeps):
            for k in ti.static(range(3)):
                self.delta_pos[j][k] = self.action_buffer[s_global][k] * self.dt
            if ti.static(self.action_dim > 3):
                for k in ti.static(range(3)):
                    self.delta_euler[j][k] = self.action_buffer[s_global][k + 3] * self.dt

    @ti.kernel
    def set_delta_pos_by_position(self, s: ti.i32, s_global: ti.i32, n_substeps: ti.i32):
        # each global step passes self.dt_global seconds, divided into n_substeps
        # pos change = action * dt
        for j in range(s * n_substeps, (s + 1) * n_substeps):
            for k in ti.static(range(3)):
                self.delta_pos[j][k] = self.action_buffer[s_global][k] / n_substeps
            if ti.static(self.action_dim > 3):
                for k in ti.static(range(3)):
                    self.delta_euler[j][k] = self.action_buffer[s_global][k + 3] / n_substeps

    @ti.kernel
    def set_action_kernel(self, s_global: ti.i32, action: ti.types.ndarray()):
        for j in ti.static(range(self.action_dim)):
            self.action_buffer[s_global][j] = action[j]

    def set_action(self, s, s_global, n_substeps, action):
        assert s_global <= self.max_action_steps_global
        assert s * n_substeps <= self.max_substeps_local
        # set actions for n_substeps ...
        if self.action_dim > 0:
            self.set_action_kernel(s_global, action)
            if self.ctrl_type == 'velocity':
                self.set_delta_pos_by_velocity(s, s_global, n_substeps)
            elif self.ctrl_type == 'position':
                self.set_delta_pos_by_position(s, s_global, n_substeps)
            else:
                raise ValueError("Unknown control type.")

    def set_action_grad(self, s, s_global, n_substeps, action):
        assert s_global <= self.max_action_steps_global
        assert s * n_substeps <= self.max_substeps_local
        if self.action_dim > 0:
            if self.ctrl_type == 'velocity':
                self.set_delta_pos_by_velocity.grad(s, s_global, n_substeps)
            elif self.ctrl_type == 'position':
                self.set_delta_pos_by_position.grad(s, s_global, n_substeps)
            else:
                raise ValueError("Unknown control type.")

    def get_action_grad(self, s, n):
        if self.action_dim > 0:
            grad = np.zeros((n + 1, self.action_dim), dtype=DTYPE_NP)
            self.get_action_v_grad_kernel(s, n, grad)
            self.get_action_p_grad_kernel(n, grad)
            return grad
        else:
            return None
