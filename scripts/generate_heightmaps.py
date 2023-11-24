import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import taichi as ti
from doma.engine.utils.mesh_ops import generate_particles_from_mesh
from vedo import Points, show, Mesh
from doma.engine.utils.misc import get_gpu_memory
import psutil
import argparse

DTYPE_NP = np.float32
DTYPE_TI = ti.f32
ti.init(arch=ti.opengl,
        # offline_cache=False, log_level=ti.TRACE,
        default_fp=DTYPE_TI, default_ip=ti.i32,
        fast_math=False, random_seed=1)


def run(args):
    height_map_res = args['hmr']
    height_map_size = 0.09  # meter
    height_map_xy_offset = (0.25 * 1000, 0.25 * 1000)
    height_map_pixel_size = height_map_size * 1000 / height_map_res
    height_map_pcd_target = ti.field(dtype=DTYPE_TI, shape=(height_map_res, height_map_res), needs_grad=True)
    down_sample_voxel_size = args['vds']

    @ti.func
    def from_xy_to_uv(x, y):
        u = (x - height_map_xy_offset[0]) / height_map_pixel_size + height_map_res / 2
        v = (y - height_map_xy_offset[1]) / height_map_pixel_size + height_map_res / 2
        return ti.floor(u, ti.i32), ti.floor(v, ti.i32)

    process = psutil.Process(os.getpid())
    script_path = os.path.dirname(os.path.realpath(__file__))
    for motion_ind in ['1', '2']:
        for agent in ['rectangle', 'round', 'cylinder']:
            for data_ind in [str(_) for _ in range(9)]:
                data_path = os.path.join(script_path, '..', f'data-motion-{motion_ind}', f'eef-{agent}')
                # hm = np.load(os.path.join(data_path, f'target_pcd_height_map-{data_ind}-res{str(height_map_res)}-vdsize{str(down_sample_voxel_size)}.npy'), allow_pickle=True)

                target_pcd_path = os.path.join(data_path, f'pcd_{data_ind}1.ply')
                obj_start_centre_real = np.load(os.path.join(data_path, f'mesh_{data_ind}0_repaired_centre.npy')).astype(DTYPE_NP)
                obj_start_centre_top_normalised = np.load(
                    os.path.join(data_path, f'mesh_{data_ind}0_repaired_normalised_centre_top.npy')).astype(DTYPE_NP)
                obj_start_initial_pos = np.array([0.25, 0.25, obj_start_centre_top_normalised[-1] + 0.01], dtype=DTYPE_NP)
                pcd_offset = - obj_start_centre_real + obj_start_initial_pos

                print(f'===> CPU memory occupied before create particles/points: {process.memory_percent()} %')
                print(f'===> GPU memory before create particles/points: {get_gpu_memory()}')

                target_pcd = o3d.io.read_point_cloud(target_pcd_path).voxel_down_sample(voxel_size=down_sample_voxel_size)
                target_pcd_points_np = np.asarray(target_pcd.points, dtype=DTYPE_NP) + pcd_offset
                target_pcd_points_np *= 1000  # convert to mm
                n_target_pcd_points = target_pcd_points_np.shape[0]
                print(f'===>  {n_target_pcd_points:7d} target points loaded.')
                target_pcd_points = ti.Vector.field(3, dtype=DTYPE_TI, shape=n_target_pcd_points)
                target_pcd_points.from_numpy(target_pcd_points_np)
                height_map_pcd_target.fill(0)

                print(f'===> CPU memory occupied after create particles/points: {process.memory_percent()} %')
                print(f'===> GPU memory after create particles/points: {get_gpu_memory()}')

                @ti.kernel
                def compute_height_map_pcd():
                    for i in range(n_target_pcd_points):
                        u, v = from_xy_to_uv(target_pcd_points[i][0], target_pcd_points[i][1])
                        ti.atomic_max(height_map_pcd_target[u, v], target_pcd_points[i][2])

                compute_height_map_pcd()
                height_map_pcd = height_map_pcd_target.to_numpy()
                plt.imshow(height_map_pcd, cmap='Greys')
                plt.savefig(
                    os.path.join(data_path, f'target_pcd_height_map-{data_ind}-res{str(height_map_res)}-vdsize{str(down_sample_voxel_size)}.npy'))
                plt.close()
                print(f'height map saved as:\n'
                      f'{os.path.join(data_path, f"target_pcd_height_map-{data_ind}-res{str(height_map_res)}-vdsize{str(down_sample_voxel_size)}.npy")}')

    # particle_density = args['pd']
    # target_particles_from_mesh_np = generate_particles_from_mesh(file=obj_end_mesh_file_path,
    #                                                              voxelize_res=1080,
    #                                                              particle_density=particle_density,
    #                                                              pos=(
    #                                                              0.25, 0.25, obj_end_centre_top_normalised[-1] + 0.01))
    # target_particles_from_mesh_np *= 1000  # convert to mm
    # target_particles_from_mesh = ti.Vector.field(3, dtype=DTYPE_TI, shape=target_particles_from_mesh_np.shape[0])
    # print(f'===>  {target_particles_from_mesh_np.shape[0]:7d} target particles generated.')
    # target_particles_from_mesh.from_numpy(target_particles_from_mesh_np.astype(DTYPE_NP))
    # height_map_particle_target = ti.field(dtype=DTYPE_TI, shape=(height_map_res, height_map_res), needs_grad=True)
    # height_map_particle_target.fill(0)
    #
    # print(f'===> CPU memory occupied after create particles/points: {process.memory_percent()} %')
    # print(f'===> GPU memory after create particles/points: {get_gpu_memory()}')
    #
    # @ti.kernel
    # def compute_height_map_particle():
    #     for i in range(target_particles_from_mesh_np.shape[0]):
    #         u, v = from_xy_to_uv(target_particles_from_mesh[i][0], target_particles_from_mesh[i][1])
    #         ti.atomic_max(height_map_particle_target[u, v], target_particles_from_mesh[i][2])
    #
    # compute_height_map_particle()
    # print(f'===> CPU memory occupied after calculation: {process.memory_percent()} %')
    # print(f'===> GPU memory after create calculation: {get_gpu_memory()}')
    #
    # height_map_particle = height_map_particle_target.to_numpy()
    # plt.imshow(height_map_particle, cmap='Greys')
    # plt.savefig(
    #     os.path.join(script_path, f'height_map_particle-data{data_ind}-res{str(height_map_res)}-density{str(particle_density)}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pd', dest='pd', default=3e7, type=float)
    parser.add_argument('--vds', dest='vds', default=0.001, type=float)
    parser.add_argument('--hmr', dest='hmr', default=32, type=int)
    arguments = vars(parser.parse_args())
    run(arguments)
