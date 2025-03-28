import numpy as np
import os
from vedo import show, Mesh, Spheres
from doma.engine.utils.mesh_ops import generate_particles_from_mesh
import pickle as pkl
import argparse


script_path = os.path.dirname(os.path.realpath(__file__))
res = 1080
particle_density = 3e7
particle_r = 0.002


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
            for pcd_index in ['0', '1']:
                mesh_path = os.path.join(data_path,
                                         f'mesh_{data_ind}{pcd_index}_repaired_normalised.obj')

                particles = generate_particles_from_mesh(mesh_path, pos=(0.25, 0.25, 0.04),
                                                         voxelize_res=res, particle_density=particle_density)
                RGB = np.zeros((len(particles), 3))
                RGB[:, 0] = particles[:, 2] / particles[:, 2].max() * 255
                RGB[:, 1] = particles[:, 2] / particles[:, 2].max() * 255
                pts = Spheres(particles, r=particle_r, c=RGB)
                print(f'Data: {mesh_path}\n'
                      f'Number of particles: {len(particles)}')
                del particles, RGB

                voxels = pkl.load(open(f"{mesh_path.replace('.obj', '')}-{res}.vox", 'rb'))
                voxels.show()
                del voxels

                mesh = Mesh(mesh_path)
                coords = mesh.points()
                coords += (0.25, 0.25, 0.04)
                coords[:, 0] += 0.06
                mesh.points(coords)

                show([pts, mesh], __doc__, axes=True).close()
                del pts, mesh


if __name__ == '__main__':
    description = 'This script is used to examine the particle system generated from mesh.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--motion', dest='motion', type=str, default='poking-1',
                        help='Name of the motion used to collect the points for reconstructing the mesh: poking-1, poking-2, poking-shifting-1, poking-shifting-2')
    parser.add_argument('--long_motion', dest='long_motion', action='store_true', help='Examine long-horizon data')
    parser.add_argument('--valid', dest='validation_data', action='store_true', help='Examine validation data')
    args = vars(parser.parse_args())
    main(args)