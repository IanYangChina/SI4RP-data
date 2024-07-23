import numpy as np
import os
import open3d as o3d
import argparse

script_path = os.path.dirname(os.path.realpath(__file__))


def main(args):
    motion = args['motion']
    assert motion in ['poking-1', 'poking-2', 'poking-shifting-1', 'poking-shifting-2']
    long_motion = args['long_motion']

    if long_motion:
        motion = 'long-horizon'
    for agent in ['rectangle', 'cylinder', 'round']:
        data_path = os.path.join(script_path, '..', '..', 'data',
                                 f'data-motion-{motion}', f'eef-{agent}')
        if args['validation_data']:
            assert not long_motion, 'Long-horizon data is for validation purposes only'
            data_path = os.path.join(script_path, '..', '..', 'data',
                                     f'data-motion-{motion}', f'eef-{agent}', 'validation_data')
        for data_ind in ['0', '1']:
            for pcd_index in ['0', '1']:  # 0: initial object pcd, 1: manipulated object pcd
                pcd_path = os.path.join(data_path, f'pcd_{data_ind}{pcd_index}.ply')
                pcd = o3d.io.read_point_cloud(pcd_path)
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                o3d.visualization.draw_geometries([pcd, frame])


if __name__ == '__main__':
    description = 'This script is used to examine the captured real-world point cloud observations.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--motion', dest='motion', type=str, default='poking-1',
                        help='Name of the motion used to collect the point clouds: poking-1, poking-2, poking-shifting-1, poking-shifting-2')
    parser.add_argument('--long_motion', dest='long-horizon', action='store_true', help='Examine long-horizon data')
    parser.add_argument('--valid', dest='validation_data', action='store_true', help='Examine validation data')
    args = vars(parser.parse_args())
    main(args)
