import os
import open3d as o3d
import pyvista as pv


world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
script_path = os.path.dirname(os.path.realpath(__file__))
pcd = o3d.io.read_point_cloud(os.path.join(script_path, '..', 'data-motion-1', 'trial-2', 'pcd_0.ply'))
o3d.visualization.draw_geometries([pcd, world_frame], width=800, height=600)
mesh = pv.read(os.path.join(script_path, '..', 'data-motion-1', 'trial-2', 'mesh_0_repaired.obj'))
pv.plot([mesh], notebook=False, window_size=[800, 600])
