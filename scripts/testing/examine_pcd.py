import numpy as np

C = np.array([0.4, 0.1, 0.1])
L = np.array([0.25, 0.25, 0.04])
up = np.array([0, 1, 0])

f = L - C
f = f / np.linalg.norm(f)  # normalize f

r = np.cross(up, f)
r = r / np.linalg.norm(r)  # normalize r
u = np.cross(f, r)
R = np.array([r, u, f])
t = -np.dot(R, C)
E = np.zeros((4, 4))
E[:3, :3] = R
E[:3, 3] = t
E[3, 3] = 1

import os
from vedo import Points, show, Mesh, Spheres, Light
import matplotlib.pyplot as plt
from doma.engine.utils.mesh_ops import generate_particles_from_mesh
import open3d as o3d
from doma.assets import asset_mesh_dir

script_path = os.path.dirname(os.path.realpath(__file__))
# bounding_box_array = np.load(os.path.join(script_path, 'reconstruction_bounding_box_array_in_base.npy'))
# bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points=o3d.utility.Vector3dVector(bounding_box_array))
# bounding_box.color = [1, 0, 0]
# world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
# table_mesh = o3d.io.read_triangle_mesh(os.path.join(asset_mesh_dir, 'raw', 'table_surface.obj'))
# table_mesh.translate([0, 0, 0.01])
table_mesh = Mesh(os.path.join(asset_mesh_dir, 'raw', 'table_surface.obj'), c=[0.1, 0.1, 0.1])
table_mesh_pts = table_mesh.points()
table_mesh_pts[:, 2] += 0.01
table_mesh.points(table_mesh_pts)

# {'pos': (0.5, 0.25, 0.2), 'color': (0.6, 0.6, 0.6)},
# {'pos': (0.5, 0.5, 1.0), 'color': (0.6, 0.6, 0.6)},
# {'pos': (0.5, 0.0, 1.0), 'color': (0.8, 0.8, 0.8)}
vedo_light_1 = Light(pos=[0.5, 0.25, 0.2], c=[0.6, 0.6, 0.6])
vedo_light_2 = Light(pos=[0.5, 0.5, 1.0], c=[0.6, 0.6, 0.6])
vedo_light_3 = Light(pos=[0.5, 0.0, 1.0], c=[0.8, 0.8, 0.8])

particle_density = 6e7

tr = 'validation'
validation_dataind_dict = {
    '2': {
        'rectangle': [4, 8],
        'round': [4, 7],
        'cylinder': [4, 7]
    },
    '4': {
        'rectangle': [2, 4],
        'round': [2, 3],
        'cylinder': [1, 3]
    }
}
for agent in ['cylinder']:
    data_path = os.path.join(script_path, '..', f'data-motion-{tr}', f'eef-{agent}')
    for data_ind in [1]:
        init_mesh_path = os.path.join(data_path, f'mesh_{data_ind}0_repaired_normalised.obj')
        obj_start_centre_real = np.load(
            os.path.join(data_path, 'mesh_' + str(data_ind) + str(0) + '_repaired_centre.npy'))
        obj_start_centre_top_normalised = np.load(
            os.path.join(data_path, 'mesh_' + str(data_ind) + str(0) + '_repaired_normalised_centre_top.npy'))
        obj_start_initial_pos = np.array([0.25, 0.25, obj_start_centre_top_normalised[-1] + 0.01])

        mesh_init = Mesh(init_mesh_path, c=(255/255, 151/255, 48/255))
        coords_init = mesh_init.points()
        coords_init += obj_start_initial_pos
        mesh_init.points(coords_init)

        init_particles_from_mesh_np = generate_particles_from_mesh(file=init_mesh_path,
                                                                   voxelize_res=1080,
                                                                   particle_density=particle_density,
                                                                   pos=obj_start_initial_pos)
        RGBA = np.ones((len(init_particles_from_mesh_np), 4)) * 255
        RGBA[:, 0] = 255
        RGBA[:, 1] = 151
        RGBA[:, 2] = 48
        init_particles_pts = Points(init_particles_from_mesh_np, r=15, c=RGBA).lighting('shiny')
        vedo_plt = show([table_mesh, init_particles_pts, vedo_light_1, vedo_light_2, vedo_light_3], __doc__,
                        axes=True, size=(640, 640), interactive=False,
                        camera={'pos': (0.4, 0.1, 0.1), 'focal_point': (0.25, 0.25, 0.04), 'viewup': (0, 0, 1)})
        arr = vedo_plt.screenshot(asarray=True)
        plt.imshow(arr)
        plt.xticks([])
        plt.yticks([])
        plt.box(False)
        # plt.show()
        plt.savefig(os.path.join(script_path, '..', f'data-motion-{tr}', f'eef-{agent}',
                                 f'prt_vis_{data_ind}0.png'), dpi=300, bbox_inches='tight', pad_inches=0)
        vedo_plt.close()

        # init_pcd_path = os.path.join(script_path, '..', f'data-motion-{tr}', f'eef-{agent}', f'pcd_{data_ind}0.ply')
        # init_pcd_o3d = o3d.io.read_point_cloud(init_pcd_path).voxel_down_sample(voxel_size=0.003)
        # init_pcd_pts_array = np.asarray(init_pcd_o3d.points)
        # RGBA = np.ones((len(init_pcd_pts_array), 4)) * 255
        # RGBA[:, 0] = 255
        # RGBA[:, 1] = 151
        # RGBA[:, 2] = 48
        init_pcd_offset = (-obj_start_centre_real + obj_start_initial_pos)
        # init_pcd_pts = Points(init_pcd_pts_array + init_pcd_offset, r=20, c=RGBA).lighting('shiny')
        # vedo_plt = show([table_mesh, init_pcd_pts, vedo_light_1, vedo_light_2, vedo_light_3], __doc__,
        #                 axes=True, size=(640, 640), interactive=False,
        #                 camera={'pos': (0.4, 0.1, 0.1), 'focal_point': (0.25, 0.25, 0.04), 'viewup': (0, 0, 1)})
        # arr = vedo_plt.screenshot(asarray=True)
        # plt.imshow(arr)
        # plt.xticks([])
        # plt.yticks([])
        # plt.box(False)
        # plt.show()
        # plt.savefig(os.path.join(script_path, '..', f'data-motion-{tr}', f'eef-{agent}',
        #                          f'pcd_vis_{data_ind}0.png'), dpi=300, bbox_inches='tight', pad_inches=0)
        # vedo_plt.close()

        # end_mesh_path = os.path.join(data_path, f'mesh_{data_ind}1_repaired.obj')
        # mesh_end = Mesh(end_mesh_path, c='g')
        # coords_end = mesh_end.points()
        # coords_end += init_pcd_offset
        # coords_end[:, 1] += 0.4
        # mesh_end.points(coords_end)

        obj_end_centre_real = np.load(os.path.join(data_path, 'mesh_' + str(data_ind) + str(1) + '_repaired_centre.npy'))
        end_normalised_mesh_path = os.path.join(data_path, f'mesh_{data_ind}1_repaired_normalised.obj')
        target_particles_from_mesh_np = generate_particles_from_mesh(file=end_normalised_mesh_path,
                                                                     voxelize_res=1080,
                                                                     particle_density=particle_density,
                                                                     pos=obj_start_initial_pos)

        mesh_target = Mesh(end_normalised_mesh_path, c=(255/255, 151/255, 48/255))
        coords_target = mesh_target.points()
        coords_target += obj_end_centre_real
        coords_target += init_pcd_offset
        mesh_target.points(coords_target)

        target_particles_from_mesh_np -= obj_start_initial_pos
        target_particles_from_mesh_np += obj_end_centre_real
        target_particles_from_mesh_np += init_pcd_offset
        RGBA = np.ones((len(target_particles_from_mesh_np), 4)) * 255
        RGBA[:, 0] = 255
        RGBA[:, 1] = 151
        RGBA[:, 2] = 48
        target_particles_pts = Points(target_particles_from_mesh_np, r=15, c=RGBA).lighting('shiny')
        vedo_plt = show([table_mesh, target_particles_pts, vedo_light_1, vedo_light_2, vedo_light_3], __doc__,
                   axes=True, size=(640, 640), interactive=False,
                   camera={'pos': (0.4, 0.1, 0.1), 'focal_point': (0.25, 0.25, 0.04), 'viewup': (0, 0, 1)})
        arr = vedo_plt.screenshot(asarray=True)
        plt.imshow(arr)
        plt.xticks([])
        plt.yticks([])
        plt.box(False)
        # plt.show()
        plt.savefig(os.path.join(script_path, '..', f'data-motion-{tr}', f'eef-{agent}',
                                 f'prt_vis_{data_ind}1.png'), dpi=300, bbox_inches='tight', pad_inches=0)
        vedo_plt.close()

        # end_pcd_path = os.path.join(script_path, '..', f'data-motion-{tr}', f'eef-{agent}', f'pcd_{data_ind}1.ply')
        # end_pcd_o3d = o3d.io.read_point_cloud(end_pcd_path).voxel_down_sample(voxel_size=0.003)
        # end_pcd_pts_array = np.asarray(end_pcd_o3d.points)
        # RGBA = np.ones((len(end_pcd_pts_array), 4)) * 255
        # RGBA[:, 0] = 255
        # RGBA[:, 1] = 151
        # RGBA[:, 2] = 48
        # pcd_end = end_pcd_pts_array + init_pcd_offset
        # end_pcd_pts = Points(pcd_end, r=20, c=RGBA).lighting('shiny')
        # vedo_plt = show([table_mesh, end_pcd_pts, vedo_light_1, vedo_light_2, vedo_light_3], __doc__,
        #            axes=True, size=(640, 640), interactive=False,
        #            camera={'pos': (0.4, 0.1, 0.1), 'focal_point': (0.25, 0.25, 0.04), 'viewup': (0, 0, 1)})
        # arr = vedo_plt.screenshot(asarray=True)
        # plt.imshow(arr)
        # plt.xticks([])
        # plt.yticks([])
        # plt.box(False)
        # plt.show()
        # plt.savefig(os.path.join(script_path, '..', f'data-motion-{tr}', f'eef-{agent}',
        #                          f'pcd_vis_{data_ind}1.png'), dpi=300, bbox_inches='tight', pad_inches=0)
        # vedo_plt.close()
