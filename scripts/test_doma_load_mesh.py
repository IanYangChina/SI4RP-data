import os
import numpy as np
import open3d as o3d
from doma.envs import ClayEnv

script_path = os.path.dirname(os.path.realpath(__file__))
mesh_file_path = os.path.join(script_path, '..', 'objects', 'pcd_to_mesh', 'mesh_0_repaired.obj')
initial_pos = (0.2, 0.25, 0.03)
env = ClayEnv(ptcl_density=1e7, horizon=500,
              mesh_file=mesh_file_path, material_id=2, voxelise_res=1080, initial_pos=initial_pos)
env.reset()
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
done = False
while not done:
    obs, reward, done, info = env.step(np.array([-0.1, -0.1, -0.1]))
    env.render(mode='human')

    if env.t == 1 or env.t == 300 or env.t == 500:
        real_pcd = o3d.io.read_point_cloud(os.path.join(script_path, '..', 'objects', 'pcd_to_mesh', 'pcd_0.ply')).paint_uniform_color([0.5, 0.5, 0.5])

        x, x_ = env.render(mode='point_cloud')
        x -= np.array(initial_pos)
        x += env.original_center
        obj_vec = o3d.utility.Vector3dVector(x)
        obj_pcd = o3d.geometry.PointCloud(obj_vec).paint_uniform_color([1, 0, 0])

        d = real_pcd.compute_point_cloud_distance(obj_pcd)
        print(np.asarray(d).mean())

        o3d.visualization.draw_geometries([
            frame,
            obj_pcd,
            real_pcd
        ], width=800, height=600)
