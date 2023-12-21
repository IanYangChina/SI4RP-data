# import os
# import pickle as pkl
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

# import trimesh
#
# # obj = pkl.load(open(os.path.join(script_path, '..', 'data-motion-1', 'eef-1', 'mesh_00_repaired_normalised-1080.vox'), 'rb'))
# # obj.show()
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

# script_path = os.path.dirname(os.path.realpath(__file__))
# bbox = np.load(os.path.join(script_path, 'reconstruction_bounding_box_array_in_base.npy'))
# print(bbox)
# bbox[4:, 2] = -0.002
# bbox[1, 0] = -0.61
# bbox[3, 0] = -0.61
# bbox[5, 0] = -0.61
# bbox[7, 0] = -0.61
# print(bbox)
# np.save(os.path.join(script_path, 'reconstruction_bounding_box_array_in_base.npy'), bbox)

print(np.random.randint(3, 5, size=20, dtype=np.int32).tolist())
exit()
