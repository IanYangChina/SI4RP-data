import copy
import os
import numpy as np
import open3d as o3d  # >= 0.14.1
from pymeshfix import _meshfix
import pyvista as pv
import trimesh
import argparse


def main(args):
    motion_name = 'poking'
    assert motion_name in ['poking', 'poking-shifting']
    motion_ind = str(1)
    assert motion_ind in ['1', '2']
    agent = 'round'
    assert agent in ['round', 'rectangle', 'cylinder']
    data_ind = '1'
    pcd_index = '1'

    script_path = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_path, '..')
    # data_path = os.path.join(script_path, '..', 'data',
    #                          f'data-motion-{motion_name}-{motion_ind}',
    #                          f'eef-{agent}')
    data_path = os.path.join(script_path, '..', 'data', 'other_mats',
                             'soil', 'pcd_to_mesh')
    extra_data = False
    if extra_data:
        data_path = os.path.join(data_path, 'extra_data')
    validation_data = False
    if validation_data:
        assert motion_ind == '2', 'Validation data is only available for motion 2.'
        data_path = os.path.join(data_path, 'validation_data')

    print(f'Processing data {data_ind} and pcd {pcd_index}.')
    pcd_path = os.path.join(data_path, 'pcd_'+data_ind+pcd_index+'.ply')
    bounding_box_array = np.load(os.path.join(script_path, 'data_generation', 'reconstruction_bounding_box_array_in_base.npy'))

    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points=o3d.utility.Vector3dVector(bounding_box_array))
    bounding_box.color = [1, 0, 0]
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    pcd = o3d.io.read_point_cloud(pcd_path).crop(bounding_box)
    o3d.io.write_point_cloud(pcd_path, pcd)
    original_pcd = copy.deepcopy(pcd)

    centre = np.asarray(pcd.points).mean(0)
    pcd = pcd.voxel_down_sample(voxel_size=0.002)  # 0.003 is a good value for downsampling

    _, ind = pcd.remove_radius_outlier(nb_points=8, radius=0.003)
    outliner = pcd.select_by_index(ind, invert=True).paint_uniform_color([1, 0, 0])
    pcd = pcd.select_by_index(ind).paint_uniform_color([0, 0.5, 0.5])

    new_points = np.asarray(pcd.points).copy()
    new_points[:, -1] = 0
    addition_pcd_vec = o3d.utility.Vector3dVector(new_points)
    addition_pcd = o3d.geometry.PointCloud(addition_pcd_vec)
    addition_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.001, max_nn=30))
    addition_pcd.orient_normals_towards_camera_location(camera_location=[centre[0], centre[1], -0.1])
    pcd = pcd + addition_pcd

    o3d.visualization.draw_geometries([pcd, outliner, world_frame, bounding_box],
                                      width=800, height=800,
                                      mesh_show_back_face=True,
                                      mesh_show_wireframe=True)

    print(original_pcd)
    print(pcd)

    radii = [0.004]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

    # o3d.visualization.draw_geometries([pcd, world_frame, bounding_box,  outliner, mesh],
    #                                   width=800, height=800,
    #                                   mesh_show_back_face=True,
    #                                   mesh_show_wireframe=True)

    mesh_path = os.path.join(data_path, 'mesh_'+data_ind+pcd_index+'.ply')
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    mesh = pv.read(mesh_path)
    os.remove(mesh_path)
    mesh_path = os.path.join(data_path, 'mesh_'+data_ind+pcd_index+'.ply')
    if args['save']:
        mesh.save(mesh_path)

    mesh_to_fix = _meshfix.PyTMesh()
    mesh_to_fix.load_file(mesh_path)
    os.remove(mesh_path)
    mesh_to_fix.join_closest_components()
    mesh_to_fix.fill_small_boundaries()

    points, faces = mesh_to_fix.return_arrays()
    mesh_centre = (points.max(0) + points.min(0)) / 2
    print(f"Original mesh centre: {mesh_centre}.")
    if args['save']:
        np.save(os.path.join(data_path, 'mesh_'+data_ind+pcd_index+'_repaired_centre.npy'), mesh_centre)
    mesh_top_centre = mesh_centre.copy()

    mesh_top_centre[-1] = points.max(0)[-1]
    print(f"Original mesh central top z: {points.max(0)[-1]}")
    print(f"Normalised mesh top z: {points.max(0)[-1] - mesh_centre[-1]}")
    if args['save']:
        np.save(os.path.join(data_path, 'mesh_'+data_ind+pcd_index+'_repaired_centre_top.npy'), mesh_top_centre)

    end_effector_target_xyz = mesh_top_centre.copy()
    end_effector_target_xyz[0] += 0.011  # real robot eef link offset = 0.008 m
    end_effector_target_xyz[-1] += 0.094  # real robot eef link offset = 0.0935 m
    print(f"Real end effector frame location: {end_effector_target_xyz}")
    repaired_mesh_path = os.path.join(data_path, 'mesh_'+data_ind+pcd_index+'_repaired.obj')
    if args['save']:
        mesh_to_fix.save_file(repaired_mesh_path)

    repaired_mesh_0 = o3d.io.read_triangle_mesh(repaired_mesh_path)
    o3d.visualization.draw_geometries([pcd, world_frame, repaired_mesh_0, bounding_box, outliner],
                                      width=800, height=800,
                                      mesh_show_back_face=True,
                                      mesh_show_wireframe=True)

    repaired_mesh_0_to_normalised = trimesh.load(repaired_mesh_path, force='mesh', skip_texture=True)
    repaired_mesh_0_to_normalised.vertices -= mesh_centre
    repaired_mesh_0_to_normalised.show()
    normalised_mesh_path = os.path.join(data_path, 'mesh_'+data_ind+pcd_index+'_repaired_normalised.obj')
    repaired_mesh_0_to_normalised.export(normalised_mesh_path, file_type='obj')
    normalised_mesh_top_centre = mesh_top_centre.copy() - mesh_centre
    print(f"Normalised mesh top centre: {normalised_mesh_top_centre}")
    print(f"Normalised mesh top centre z: {repaired_mesh_0_to_normalised.vertices.max(0)[-1]}")
    if args['save']:
        np.save(os.path.join(data_path, 'mesh_'+data_ind+pcd_index+'_repaired_normalised_centre_top.npy'), normalised_mesh_top_centre)


if __name__ == '__main__':
    description = ("This script generates meshes from point clouds. "
                   "For different point clouds one may need to fine-tune the radii used by the ball-pivoting algorithm to achieve better meshing details.")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--save', dest='save', default=True, action='store_true',
                        help="Save the mesh. Not recommended as this alternates the files uploaded by the authors.")
    arguments = vars(parser.parse_args())
    main(arguments)
