import os
import pickle as pkl
script_path = os.path.dirname(os.path.realpath(__file__))
# import open3d as o3d
# import pyvista as pv
#
#
# world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
# script_path = os.path.dirname(os.path.realpath(__file__))
# pcd = o3d.io.read_point_cloud(os.path.join(script_path, '..', 'data-motion-1', 'trial-2', 'pcd_0.ply'))
# o3d.visualization.draw_geometries([pcd, world_frame], width=800, height=600)
# mesh = pv.read(os.path.join(script_path, '..', 'data-motion-1', 'trial-2', 'mesh_0_repaired.obj'))
# pv.plot([mesh], notebook=False, window_size=[800, 600])

import trimesh

# obj = pkl.load(open(os.path.join(script_path, '..',
#                                  'data-motion-3',
#                                  'eef-round', 'mesh_01_repaired_normalised-1080.vox'), 'rb'))
# obj.show()
# exit()
#
#
# import taichi as ti
# ti.init(arch=ti.vulkan)
# a = ti.Vector.field(n=3, m=3, dtype=ti.f32, shape=())
# b = ti.Vector.field(n=3, m=3, dtype=ti.f32, shape=())
# a.fill(1)
# b.fill(2)
# print(a[None] / b[None])

import numpy as np

v_buffer = 3.0064564e+26
grad = 9.111189305277442e+19
v_t = np.array(0.999 * v_buffer + (1 - 0.999) * (grad * grad)).astype(np.float32)
print(not (np.isnan(v_t) or np.isinf(v_t)))
exit()

import os
import matplotlib.pyplot as plt
# x = np.linspace(0, 1, 100)
# y = np.linspace(0, 1, 100)
# z = np.linspace(0, 1, 100)
# X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
# print(X.shape, Y.shape, Z.shape)
#
# p = np.stack(np.meshgrid(x, y, z, indexing='ij'), -1)
# print(p.shape)

bbox = np.load(os.path.join(script_path, 'reconstruction_bounding_box_array_in_base.npy'))
print(bbox)
# bbox[0, 1] = 0.06
# bbox[1, 1] = 0.06
# bbox[2, 1] = -0.06
# bbox[3, 1] = -0.06
# bbox[4, 1] = 0.06
# bbox[5, 1] = 0.06
# bbox[6, 1] = -0.06
# bbox[7, 1] = -0.06
bbox[:4, 2] = 0.06
print(bbox)
np.save(os.path.join(script_path, 'reconstruction_bounding_box_array_in_base.npy'), bbox)

# print(np.random.randint(3, 5, size=20, dtype=np.int32).tolist())
exit()
