import taichi as ti
from doma.engine.configs.macros import EPS

@ti.func
def normalize(v, eps=EPS):
    return v / (v.norm(eps))


@ti.func
def transform_by_quat_ti(v, quat):
    qvec = ti.Vector([quat[1], quat[2], quat[3]])
    uv = qvec.cross(v)
    uuv = qvec.cross(uv)
    return v + 2 * (quat[0] * uv + uuv)


@ti.func
def inv_transform_by_quat_ti(v, quat):
    return transform_by_quat_ti(v, inv_quat(quat))


@ti.func
def transform_by_T_ti(pos, T, dtype):
    new_pos = ti.Vector([pos[0], pos[1], pos[2], 1.0], dt=dtype)
    new_pos = T @ new_pos
    return new_pos[:3]


@ti.func
def transform_by_trans_quat_ti(pos, trans, quat):
    return transform_by_quat_ti(pos, quat) + trans


@ti.func
def inv_transform_by_trans_quat_ti(pos, trans, quat):
    return transform_by_quat_ti(pos - trans, inv_quat(quat))


@ti.func
def qmul(q, r):
    terms = r.outer_product(q)
    w = terms[0, 0] - terms[1, 1] - terms[2, 2] - terms[3, 3]
    x = terms[0, 1] + terms[1, 0] - terms[2, 3] + terms[3, 2]
    y = terms[0, 2] + terms[1, 3] + terms[2, 0] - terms[3, 1]
    z = terms[0, 3] - terms[1, 2] + terms[2, 1] + terms[3, 0]
    out = ti.Vector([w, x, y, z])
    return out / ti.sqrt(out.dot(out))  # normalize it to prevent some unknown NaN problems.


@ti.func
def w2quat(axis_angle, dtype):
    w = axis_angle.norm(EPS)
    out = ti.Vector.zero(dt=dtype, n=4)

    v = (axis_angle / w) * ti.sin(w / 2)
    out[0] = ti.cos(w / 2)
    out[1] = v[0]
    out[2] = v[1]
    out[3] = v[2]

    return out


@ti.func
def inv_quat(quat):
    return ti.Vector([quat[0], -quat[1], -quat[2], -quat[3]]).normalized()


@ti.func
def inv_trans_ti(pos_A, trans_B_to_A, rot_B_to_A):
    return rot_B_to_A.inverse() @ (pos_A - trans_B_to_A)


@ti.func
def trans_ti(pos_B, trans_B_to_A, rot_B_to_A):
    return rot_B_to_A.inverse() @ pos_B + trans_B_to_A
