import numpy as np
import os
from vedo import Points, show, Mesh, Spheres
from doma.engine.utils.mesh_ops import generate_particles_from_mesh
import open3d as o3d

script_path = os.path.dirname(os.path.realpath(__file__))
bounding_box_array = np.load(os.path.join(script_path, 'reconstruction_bounding_box_array_in_base.npy'))
bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points=o3d.utility.Vector3dVector(bounding_box_array))
bounding_box.color = [1, 0, 0]

# Data: /home/xintong/Documents/PyProjects/SI4RP-data/scripts/../data-motion-2/eef-round/pcd_31.ply

tr = '2'
for agent in ['round']:
    data_path = os.path.join(script_path, '..', f'data-motion-{tr}', f'eef-{agent}')
    for data_ind in ['1', '3', '4', '5', '6', '7', '8']:
        init_mesh_path = os.path.join(data_path, f'mesh_{data_ind}0_repaired_normalised.obj')
        obj_start_centre_real = np.load(os.path.join(data_path, 'mesh_' + data_ind + str(0) + '_repaired_centre.npy'))
        obj_start_centre_top_normalised = np.load(
            os.path.join(data_path, 'mesh_' + data_ind + str(0) + '_repaired_normalised_centre_top.npy'))
        obj_start_initial_pos = np.array([0.25, 0.25, obj_start_centre_top_normalised[-1] + 0.01])
        
        mesh_init = Mesh(init_mesh_path, c='r')
        coords_init = mesh_init.points()
        coords_init += obj_start_initial_pos
        coords_init[:, 1] += 0.1
        mesh_init.points(coords_init)

        init_particles_from_mesh_np = generate_particles_from_mesh(file=init_mesh_path,
                                                                     voxelize_res=1080,
                                                                     particle_density=2e7,
                                                                     pos=obj_start_initial_pos)
        init_particles_from_mesh_np[:, 1] += 0.2
        init_particles_pts = Points(init_particles_from_mesh_np, r=12, c='r')

        init_pcd_path = os.path.join(script_path, '..', f'data-motion-{tr}', f'eef-{agent}', f'pcd_{data_ind}0.ply')
        init_pcd_pts = Points(init_pcd_path, r=12)
        init_pcd_offset = (-obj_start_centre_real + obj_start_initial_pos)
        pcd_init = init_pcd_pts.points() + init_pcd_offset
        pcd_init[:, 1] += 0.3
        init_pcd_pts = Points(pcd_init, r=12, c='r')

        end_mesh_path = os.path.join(data_path, f'mesh_{data_ind}1_repaired.obj')
        obj_end_centre_real = np.load(os.path.join(data_path, 'mesh_' + data_ind + str(1) + '_repaired_centre.npy'))
        mesh_end = Mesh(end_mesh_path, c='g')
        coords_end = mesh_end.points()
        coords_end += init_pcd_offset
        coords_end[:, 1] += 0.4
        mesh_end.points(coords_end)

        end_normalised_mesh_path = os.path.join(data_path, f'mesh_{data_ind}1_repaired_normalised.obj')
        target_particles_from_mesh_np = generate_particles_from_mesh(file=end_normalised_mesh_path,
                                                                     voxelize_res=1080,
                                                                     particle_density=2e7,
                                                                     pos=obj_start_initial_pos)
        target_particles_from_mesh_np -= obj_start_initial_pos
        target_particles_from_mesh_np += obj_end_centre_real
        target_particles_from_mesh_np += init_pcd_offset
        target_particles_from_mesh_np[:, 1] += 0.5
        target_pparticles_pts = Points(target_particles_from_mesh_np, r=12, c='g')

        end_pcd_path = os.path.join(script_path, '..', f'data-motion-{tr}', f'eef-{agent}', f'pcd_{data_ind}1.ply')
        end_pcd_pts = Points(end_pcd_path, r=12)
        pcd_end = end_pcd_pts.points() + init_pcd_offset
        pcd_end[:, 1] += 0.6
        end_pcd_pts = Points(pcd_end, r=12, c='g')

        show([mesh_init, init_pcd_pts, init_particles_pts,
              mesh_end, end_pcd_pts, target_pparticles_pts], __doc__, axes=True).close()

        # _, ind = pcd.remove_radius_outlier(nb_points=3, radius=0.001)
        # outliner = pcd.select_by_index(ind, invert=True).paint_uniform_color([1, 0, 0])
        # pcd = pcd.select_by_index(ind).paint_uniform_color([0, 0.5, 0.5])
        #
        # o3d.visualization.draw_geometries([pcd, bounding_box, outliner],
        #                                   width=800, height=800)

        # pcd = pcd.crop(bounding_box)
        # o3d.io.write_point_cloud(pcd_path, pcd)