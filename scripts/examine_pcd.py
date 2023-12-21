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
    for data_ind in ['3']:
        for pcd_index in ['1']:
            pcd_path = os.path.join(script_path, '..', f'data-motion-{tr}', f'eef-{agent}',
                                     f'pcd_{data_ind}{pcd_index}.ply')
            print(f"Data: {pcd_path}")
            pcd = o3d.io.read_point_cloud(pcd_path)

            o3d.visualization.draw_geometries([pcd, bounding_box],
                                              width=800, height=800)

            # _, ind = pcd.remove_radius_outlier(nb_points=3, radius=0.001)
            # outliner = pcd.select_by_index(ind, invert=True).paint_uniform_color([1, 0, 0])
            # pcd = pcd.select_by_index(ind).paint_uniform_color([0, 0.5, 0.5])
            #
            # o3d.visualization.draw_geometries([pcd, bounding_box, outliner],
            #                                   width=800, height=800)

            # pcd = pcd.crop(bounding_box)
            # o3d.io.write_point_cloud(pcd_path, pcd)