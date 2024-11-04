import argparse
import os
from vedo import Points, show, Mesh, Spheres, Light
import matplotlib.pyplot as plt
from doma.engine.utils.mesh_ops import generate_particles_from_mesh
import open3d as o3d
from doma.assets import asset_mesh_dir
import numpy as np

script_path = os.path.dirname(os.path.realpath(__file__))
table_mesh = Mesh(os.path.join(asset_mesh_dir, 'raw', 'table_surface.obj'), c=[0.1, 0.1, 0.1])
table_mesh_pts = table_mesh.points()
table_mesh_pts[:, 2] += 0.01
table_mesh.points(table_mesh_pts)

vedo_light_1 = Light(pos=[0.5, 0.25, 0.2], c=[0.6, 0.6, 0.6])
vedo_light_2 = Light(pos=[0.5, 0.5, 1.0], c=[0.6, 0.6, 0.6])
vedo_light_3 = Light(pos=[0.5, 0.0, 1.0], c=[0.8, 0.8, 0.8])


def main(args):
    # motion = args['motion']
    # assert motion in ['poking-1', 'poking-2', 'poking-shifting-1', 'poking-shifting-2']
    # long_motion = args['long_motion']
    #
    # if long_motion:
    #     motion = 'long-horizon'
    # for agent in ['rectangle', 'cylinder', 'round']:
    #     data_path = os.path.join(script_path, '..', '..', 'data',
    #                              f'data-motion-{motion}', f'eef-{agent}')
    #     if args['validation_data']:
    #         assert not long_motion, 'Long-horizon data is for validation purposes only'
    #         data_path = os.path.join(script_path, '..', '..', 'data',
    #                                  f'data-motion-{motion}', f'eef-{agent}', 'validation_data')
    #     for data_ind in ['0', '1']:
    #         for pcd_index in ['0', '1']:  # 0: initial object pcd, 1: manipulated object pcd
    #             pcd_path = os.path.join(data_path, f'pcd_{data_ind}{pcd_index}.ply')
    #             pcd = o3d.io.read_point_cloud(pcd_path)
    #             frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    #             o3d.visualization.draw_geometries([pcd, frame])

    mat = 'soil'
    data_path = os.path.join(script_path, '..', '..', 'data', 'other_mats', mat, 'long-motion-validation')
    # for pcd_index in ['0', '1']:
    #     pcd_path = os.path.join(data_path, f'pcd_0{pcd_index}.ply')
    #     pcd = o3d.io.read_point_cloud(pcd_path)
    #     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    #     o3d.visualization.draw_geometries([pcd, frame])

    obj_start_centre_real = np.load(
        os.path.join(data_path, 'mesh_01_repaired_centre.npy'))
    obj_start_centre_top_normalised = np.load(
        os.path.join(data_path, 'mesh_01_repaired_normalised_centre_top.npy'))
    obj_start_initial_pos = np.array([0.25, 0.25, obj_start_centre_top_normalised[-1] + 0.01])

    init_pcd_offset = (-obj_start_centre_real + obj_start_initial_pos)

    obj_end_centre_real = np.load(
        os.path.join(data_path, 'mesh_01_repaired_centre.npy'))
    end_normalised_mesh_path = os.path.join(data_path, f'mesh_01_repaired_normalised.obj')
    target_particles_from_mesh_np = generate_particles_from_mesh(file=end_normalised_mesh_path,
                                                                 voxelize_res=1080,
                                                                 particle_density=4e7,
                                                                 pos=obj_start_initial_pos)

    mesh_target = Mesh(end_normalised_mesh_path, c=(255 / 255, 151 / 255, 48 / 255))
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
                    camera={'pos': (0.42, -0.02, 0.1), 'focal_point': (0.25, 0.25, 0.03), 'viewup': (0, 0, 1)})
    arr = vedo_plt.screenshot(asarray=True)
    plt.imshow(arr)
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    # plt.show()
    plt.savefig(os.path.join(script_path, '..', '..', 'figures', 'result-figs', 'other_mats',
                             f'{mat}_prt_vis_01.png'), dpi=300, bbox_inches='tight', pad_inches=0)
    vedo_plt.close()

    end_pcd_path = os.path.join(data_path, f'pcd_01.ply')
    end_pcd_o3d = o3d.io.read_point_cloud(end_pcd_path).voxel_down_sample(voxel_size=0.003)
    end_pcd_pts_array = np.asarray(end_pcd_o3d.points)
    RGBA = np.ones((len(end_pcd_pts_array), 4)) * 255
    RGBA[:, 0] = 255
    RGBA[:, 1] = 151
    RGBA[:, 2] = 48
    pcd_end = end_pcd_pts_array + init_pcd_offset
    end_pcd_pts = Points(pcd_end, r=20, c=RGBA).lighting('shiny')
    vedo_plt = show([table_mesh, end_pcd_pts, vedo_light_1, vedo_light_2, vedo_light_3], __doc__,
                    axes=True, size=(640, 640), interactive=False,
                    camera={'pos': (0.42, -0.02, 0.1), 'focal_point': (0.25, 0.25, 0.03), 'viewup': (0, 0, 1)})
    arr = vedo_plt.screenshot(asarray=True)
    plt.imshow(arr)
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    # plt.show()
    plt.savefig(os.path.join(script_path, '..', '..', 'figures', 'result-figs', 'other_mats',
                             f'{mat}_pcd_vis_01.png'), dpi=300, bbox_inches='tight', pad_inches=0)
    vedo_plt.close()


if __name__ == '__main__':
    description = 'This script is used to examine the captured real-world point cloud observations.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--motion', dest='motion', type=str, default='poking-1',
                        help='Name of the motion used to collect the point clouds: poking-1, poking-2, poking-shifting-1, poking-shifting-2')
    parser.add_argument('--long_motion', dest='long-horizon', action='store_true', help='Examine long-horizon data')
    parser.add_argument('--valid', dest='validation_data', action='store_true', help='Examine validation data')
    arguments = vars(parser.parse_args())
    main(arguments)