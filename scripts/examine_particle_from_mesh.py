import numpy as np
import os
from vedo import Points, show, Mesh, Spheres
from doma.engine.utils.mesh_ops import generate_particles_from_mesh
import trimesh
import pickle as pkl

script_path = os.path.dirname(os.path.realpath(__file__))
agent = 'rectangle'  # 'round' 'cylinder'
tr = '2'
data_ind = '1'
res = 1080
particle_density = 3e7
particle_r = 0.0015

# 21, 41
for data_ind in ['2', '4']:
    for pcd_index in ['1']:
        mesh_path = os.path.join(script_path, '..', f'data-motion-{tr}', f'eef-{agent}',
                                 f'mesh_{data_ind}{pcd_index}_repaired_normalised.obj')

        particles = generate_particles_from_mesh(mesh_path, pos=(0.25, 0.25, 0.04),
                                                      voxelize_res=res, particle_density=particle_density)
        RGBA = np.zeros((len(particles), 3))
        RGBA[:, 0] = particles[:, 2] / particles[:, 2].max() * 255
        RGBA[:, 1] = particles[:, 2] / particles[:, 2].max() * 255
        pts = Spheres(particles, r=particle_r, c=RGBA)
        print(f'Data: {mesh_path}\n'
              f'Number of particles: {len(particles)}')
        del particles, RGBA

        # voxels = pkl.load(open(f"{mesh_path.replace('.obj', '')}-{res}.vox", 'rb'))
        # voxels.show()
        # del voxels

        mesh = Mesh(mesh_path)
        coords = mesh.points()
        coords += (0.25, 0.25, 0.04)
        coords[:, 0] += 0.06
        mesh.points(coords)

        show([pts, mesh], __doc__, axes=True).close()
        del pts, mesh
