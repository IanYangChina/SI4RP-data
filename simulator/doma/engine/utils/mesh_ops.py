import os
import trimesh
import numpy as np
import pickle as pkl
from doma.assets import asset_mesh_dir
from mesh_to_sdf import mesh_to_sdf


def get_raw_mesh_path(file):
    assert file.endswith('.obj')
    return os.path.join(asset_mesh_dir, 'raw', file)


def get_processed_mesh_path(file, file_vis):
    assert file.endswith('.obj') and file_vis.endswith('.obj')
    processed_file = f"{file.replace('.obj', '')}-{file_vis.replace('.obj', '')}.obj"
    processed_file_path = os.path.join(asset_mesh_dir, 'processed', processed_file)
    return processed_file_path


def get_processed_sdf_path(file, sdf_res):
    assert file.endswith('.obj')
    processed_sdf = f"{file.replace('.obj', '')}-{sdf_res}.sdf"
    processed_sdf_path = os.path.join(asset_mesh_dir, 'processed', processed_sdf)
    return processed_sdf_path


def get_voxelized_mesh_path(file, voxelize_res):
    assert file.endswith('.obj')
    return os.path.join(asset_mesh_dir, 'voxelized',
                        f"{file.replace('.obj', '')}-{voxelize_res}.vox")


def load_mesh(file):
    return trimesh.load(file, force='mesh', skip_texture=True)


def normalize_mesh(mesh, mesh_actual=None, scale=True):
    '''
    Normalize mesh_dict to [-0.5, 0.5] using size of mesh_dict_actual.
    '''
    if mesh_actual is None:
        mesh_actual = mesh

    scale_ratio = (mesh_actual.vertices.max(0) - mesh_actual.vertices.min(0)).max()
    center = (mesh_actual.vertices.max(0) + mesh_actual.vertices.min(0)) / 2
    normalized_mesh = mesh.copy()
    normalized_mesh.vertices -= center
    if scale:
        normalized_mesh.vertices /= scale_ratio
    return normalized_mesh


def scale_mesh(mesh, scale):
    scale = np.array(scale)
    return trimesh.Trimesh(
        vertices=mesh.vertices * scale,
        faces=mesh.faces,
    )


def cleanup_mesh(mesh):
    '''
    Retain only mesh's vertices, faces, and normals.
    '''
    return trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_normals=mesh.vertex_normals,
        face_normals=mesh.face_normals,
    )


def compute_sdf_data(mesh, res):
    '''
    Convert mesh to sdf voxels and a transformation matrix from mesh frame to voxel frame.
    '''
    scan_count = int(res / 64 * 100)
    scan_resolution = 400
    center = (mesh.vertices.max(0) + mesh.vertices.min(0)) / 2
    voxels_radius = (mesh.vertices.max(0) - mesh.vertices.min(0)).max() * 1.2 / 2
    voxel_radius_lower = center - voxels_radius
    voxel_radius_upper = center + voxels_radius
    x = np.linspace(voxel_radius_lower[0], voxel_radius_upper[0], num=res)
    y = np.linspace(voxel_radius_lower[1], voxel_radius_upper[1], num=res)
    z = np.linspace(voxel_radius_lower[2], voxel_radius_upper[2], num=res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    query_points = np.stack([X, Y, Z], axis=-1).reshape((-1, 3))

    voxels = mesh_to_sdf(mesh, query_points, scan_count=scan_count, scan_resolution=scan_resolution,
                         sign_method='depth',
                         # meshes with holes may not work well with depth-based sign computation
                         # normal-based sign computation generate sdf artifacts
                         # see FAQ in https://github.com/marian42/mesh_to_sdf
                         normal_sample_count=11)
    # note that the sdf is computed after the mesh is centralised
    voxels = voxels.reshape([res, res, res])

    T_mesh_to_voxels = np.eye(4)
    T_mesh_to_voxels[:3, :3] *= (res - 1) / (voxels_radius * 2)
    T_mesh_to_voxels[:3, 3] = (voxels_radius - center) * (res - 1) / (2 * voxels_radius)

    sdf_data = {
        'voxels': voxels,
        'T_mesh_to_voxels': T_mesh_to_voxels,
    }
    return sdf_data


def voxelize_mesh(file, voxelised_mesh_file, res, normalize=False, scale=False, save_voxels=True):
    if not os.path.exists(voxelised_mesh_file):
        print(f'===> Voxelizing mesh {file}.')
        raw_mesh = load_mesh(file)
        if normalize:
            raw_mesh = normalize_mesh(raw_mesh, scale=scale)
            raw_mesh = cleanup_mesh(raw_mesh)
        voxelized_mesh = raw_mesh.voxelized(pitch=1.0 / res).fill()
        if save_voxels:
            pkl.dump(voxelized_mesh, open(voxelised_mesh_file, 'wb'))
        print(f'===> Voxelized mesh saved as {voxelised_mesh_file}.')
    else:
        voxelized_mesh = pkl.load(open(voxelised_mesh_file, 'rb'))
    return voxelized_mesh


def generate_particles_from_mesh(file,
                                 pos=(0.5, 0.5, 0.5), scale=(1.0, 1.0, 1.0),
                                 voxelize_res=128, particle_density=1e6):
    raw_file_path = get_raw_mesh_path(file)
    voxelized_file_path = get_voxelized_mesh_path(file, voxelize_res)
    voxels = voxelize_mesh(raw_file_path, voxelized_file_path, voxelize_res)

    # sample a cube around pos
    scale = np.array(scale)
    pos = np.array(pos)
    cube_lower = pos - scale * 0.5
    cube_upper = pos + scale * 0.5
    size = cube_upper - cube_lower
    n_x = int(round(size[0] * np.cbrt(particle_density)))
    n_y = int(round(size[1] * np.cbrt(particle_density)))
    n_z = int(round(size[2] * np.cbrt(particle_density)))
    x = np.linspace(cube_lower[0], cube_upper[0], n_x + 1)
    y = np.linspace(cube_lower[1], cube_upper[1], n_y + 1)
    z = np.linspace(cube_lower[2], cube_upper[2], n_z + 1)
    particles = np.stack(np.meshgrid(x, y, z, indexing='ij'), -1).reshape((-1, 3))
    particles = particles[voxels.is_filled((particles - pos) / scale)]

    return particles
