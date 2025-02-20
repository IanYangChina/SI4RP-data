import numpy as np
from scipy.spatial.transform import Rotation


def scale_to_T(scale):
    T_scale = np.eye(4, dtype=scale.dtype)
    T_scale[[0, 1, 2], [0, 1, 2]] = scale
    return T_scale


def trans_quat_to_T(trans=None, quat=None):
    if trans is not None:
        dtype = trans.dtype
    else:
        dtype = quat.dtype

    T = np.eye(4, dtype=dtype)
    if trans is not None:
        T[:3, 3] = trans
    if quat is not None:
        T[:3, :3] = Rotation.from_quat(xyzw_from_wxyz(quat)).as_matrix()

    return T


def transform_by_T_np(pos, T):
    if len(pos.shape) == 2:
        assert pos.shape[1] == 3
        new_pos = np.hstack([pos, np.ones_like(pos[:, :1])]).T
        new_pos = (T @ new_pos).T
        new_pos = new_pos[:, :3]

    elif len(pos.shape) == 1:
        assert pos.shape[0] == 3
        new_pos = np.append(pos, 1)
        new_pos = T @ new_pos
        new_pos = new_pos[:3]

    else:
        assert False

    return new_pos


def transform_by_trans_quat_np(pos, trans=None, quat=None):
    return transform_by_quat_np(pos, quat) + trans


def transform_by_quat_np(v, quat):
    qvec = quat[1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2 * (quat[0] * uv + uuv)


def xyzw_to_wxyz(xyzw):
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def xyzw_from_wxyz(wxyz):
    return np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])


def compute_camera_angle(camera_pos, camera_lookat):
    camera_dir = np.array(camera_lookat) - np.array(camera_pos)

    # rotation around vertical (y) axis
    angle_x = np.arctan2(-camera_dir[0], -camera_dir[2])

    # rotation w.r.t horizontal plane
    angle_y = np.arctan2(camera_dir[1], np.linalg.norm([camera_dir[0], camera_dir[2]]))

    angle_z = 0.0

    return np.array([angle_x, angle_y, angle_z])


def construct_homogeneous_transform_matrix(translation, orientation):
    translation = np.array(translation).reshape((3, 1))  # xyz
    if len(orientation) == 4:
        rotation = Rotation.from_quat(np.array(orientation))  # xyzw
    else:
        assert len(orientation) == 3, 'orientation should be a quaternion or 3 axis angles'
        rotation = np.radians(np.array(orientation).astype("float"))  # xyz in radians
        rotation = Rotation.from_euler('xyz', rotation).as_matrix()
    transformation = np.append(rotation, translation, axis=1)
    transformation = np.append(transformation, np.array([[0, 0, 0, 1]]), axis=0)
    return transformation