import copy
import os
import numpy as np
import open3d as o3d  # >= 0.14.1
from pymeshfix import _meshfix
import pyvista as pv

motion_ind = str(1)
pcd_index = str(2)
script_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_path, '..', 'data-motion-'+motion_ind, 'pcd_to_mesh')

pcd_path = os.path.join(data_path, 'pcd_'+pcd_index+'.ply')
bounding_box_array = np.load(os.path.join(script_path, 'reconstruction_bounding_box_array_in_base.npy'))

bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points=o3d.utility.Vector3dVector(bounding_box_array))
bounding_box.color = [1, 0, 0]
world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
pcd = o3d.io.read_point_cloud(pcd_path)
original_pcd = copy.deepcopy(pcd)
_, ind = pcd.remove_radius_outlier(nb_points=7, radius=0.005)
outliner = pcd.select_by_index(ind, invert=True).paint_uniform_color([1, 0, 0])
pcd = pcd.select_by_index(ind).paint_uniform_color([0, 0.5, 0.5])

o3d.visualization.draw_geometries([original_pcd, world_frame, bounding_box, outliner],
                                  width=800, height=800,
                                  mesh_show_back_face=True,
                                  mesh_show_wireframe=True)

pcd = pcd.voxel_down_sample(voxel_size=0.002)  # 0.003 is a good value for downsampling

radii = [0.003]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

mesh_path = os.path.join(data_path, 'mesh_'+pcd_index+'.obj')
o3d.io.write_triangle_mesh(mesh_path, mesh)
mesh = pv.read(mesh_path)
os.remove(mesh_path)
mesh_path = os.path.join(data_path, 'mesh_'+pcd_index+'.ply')
mesh.save(mesh_path)

mesh_to_fix = _meshfix.PyTMesh()
mesh_to_fix.load_file(mesh_path)
os.remove(mesh_path)
mesh_to_fix.fill_small_boundaries()
points, faces = mesh_to_fix.return_arrays()
mesh_centre = (points.max(0) + points.min(0)) / 2
print(mesh_centre)
np.save(os.path.join(data_path, 'mesh_'+pcd_index+'_repaired_centre.npy'), mesh_centre)
mesh_top_centre = mesh_centre.copy()
mesh_top_centre[-1] = points.max(0)[-1]
np.save(os.path.join(data_path, 'mesh_'+pcd_index+'_repaired_top_centre.npy'), mesh_top_centre)

end_effector_target_xyz = mesh_top_centre.copy()
end_effector_target_xyz[-1] += 0.0935  # real robot eef link offset = 0.0935 m
print(end_effector_target_xyz)
repaired_mesh_path = os.path.join(data_path, 'mesh_'+pcd_index+'_repaired.obj')
mesh_to_fix.save_file(repaired_mesh_path)

repaired_mesh_0 = o3d.io.read_triangle_mesh(repaired_mesh_path)
o3d.visualization.draw_geometries([pcd, world_frame, repaired_mesh_0, bounding_box, outliner],
                                  width=800, height=800,
                                  mesh_show_back_face=True,
                                  mesh_show_wireframe=True)
