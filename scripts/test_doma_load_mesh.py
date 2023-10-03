import os
import numpy as np
import open3d as o3d
from doma.envs import ClayEnv
import time

script_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(script_path, '..', 'data-motion-1', 'trial-1')
pcd_ind = str(1)
pcd_ind_ = str(2)
mesh_file_path = os.path.join(data_path, 'mesh_'+pcd_ind+'_repaired_normalised.obj')
centre_real = np.load(os.path.join(data_path, 'mesh_'+pcd_ind+'_repaired_centre.npy'))
centre_top_normalised = np.load(os.path.join(data_path, 'mesh_'+pcd_ind+'_repaired_normalised_centre_top.npy'))
initial_pos = (0.25, 0.25, centre_top_normalised[-1]+0.01)

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

horizon = int(0.03 / 0.002)
env = ClayEnv(ptcl_density=3e7, horizon=100,
              mesh_file=mesh_file_path, material_id=2, voxelise_res=1080, initial_pos=initial_pos)
env.reset()

env.mpm_env.apply_agent_action_p(np.array([0.25, 0.25, 2*centre_top_normalised[-1]+0.01, 0, 0, 45]))
v = 0.045 / 0.03  # 1.5 m/s
trajectory_1 = np.zeros(shape=(horizon, env.mpm_env.agent.action_dim))
trajectory_1[:5, 2] = -v
trajectory_1[5:, 2] = v

done = False
while not done:
    if env.t < 15:
        a = trajectory_1[env.t]
    else:
        a = np.zeros(shape=(env.mpm_env.agent.action_dim,))
    obs, reward, done, info = env.step(a)
    env.render(mode='human')
    time.sleep(0.2)
    if env.t == 1 or env.t == 16:
        real_pcd_0 = o3d.io.read_point_cloud(os.path.join(data_path, 'pcd_'+pcd_ind+'.ply')).paint_uniform_color([0.5, 0.5, 0.5])
        real_pcd_1 = o3d.io.read_point_cloud(os.path.join(data_path, 'pcd_'+pcd_ind_+'.ply')).paint_uniform_color([0.5, 0.5, 0.5])
        # real_pcd_2 = o3d.io.read_point_cloud(os.path.join(data_path, 'pcd_0.ply')).paint_uniform_color([0.5, 0.5, 0.5])

        x, x_ = env.render(mode='point_cloud')
        x -= np.array(initial_pos)
        x += centre_real
        obj_vec = o3d.utility.Vector3dVector(x)
        obj_pcd = o3d.geometry.PointCloud(obj_vec).paint_uniform_color([1, 0, 0])

        x_[:, :-1] -= np.array(initial_pos[:-1])
        x_[:, :-1] += centre_real[:-1]
        eff_vec = o3d.utility.Vector3dVector(x_)
        eff_pcd = o3d.geometry.PointCloud(eff_vec).paint_uniform_color([0, 1, 0])

        d_rs_0 = real_pcd_0.compute_point_cloud_distance(obj_pcd)
        d_sr_0 = obj_pcd.compute_point_cloud_distance(real_pcd_0)
        print(np.asarray(d_rs_0).mean(), np.asarray(d_sr_0).mean())
        d_rs_1 = real_pcd_1.compute_point_cloud_distance(obj_pcd)
        d_sr_1 = obj_pcd.compute_point_cloud_distance(real_pcd_1)
        print(np.asarray(d_rs_1).mean(), np.asarray(d_sr_1).mean())

        o3d.visualization.draw_geometries([
            frame,
            eff_pcd,
            obj_pcd,
            real_pcd_1
        ], width=800, height=600)
