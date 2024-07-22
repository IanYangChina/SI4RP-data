import numpy as np
import os
import open3d as o3d

script_path = os.path.dirname(os.path.realpath(__file__))
agent = 'round'
assert agent in ['round', 'cylinder', 'rectangle']
motion = 'poking-1'
assert motion in ['poking-1', 'poking-2', 'poking-shifting-1', 'poking-shifting-2']
res = 1080
particle_density = 3e7
particle_r = 0.002

for data_ind in ['0', '1']:
    for pcd_index in ['0', '1']:  # 0: initial object pcd, 1: manipulated object pcd
        pcd_path = os.path.join(script_path, '..', '..', 'data',
                                 f'data-motion-{motion}', f'eef-{agent}',
                                 f'pcd_{data_ind}{pcd_index}.ply')
        pcd = o3d.io.read_point_cloud(pcd_path)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, frame])
