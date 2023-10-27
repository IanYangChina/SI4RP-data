import numpy as np
import os
from vedo import Points, show, Mesh
from doma.engine.utils.mesh_ops import generate_particles_from_mesh

script_path = os.path.dirname(os.path.realpath(__file__))
agent = 'rectangle'  # 'round' 'cylinder'
tr = '1'
data_ind = '6'
res = 1080
particle_density = 4e7

init_mesh_path = os.path.join(script_path, '..', f'data-motion-{tr}', f'eef-{agent}',
                         f'mesh_{data_ind}0_repaired_normalised.obj')

init_particles = generate_particles_from_mesh(init_mesh_path, pos=(0.25, 0.25, 0.04), voxelize_res=res, particle_density=particle_density)
RGBA = np.zeros((len(init_particles), 4))
RGBA[:, 0] = init_particles[:, 2] / init_particles[:, 2].max() * 255
RGBA[:, 1] = init_particles[:, 2] / init_particles[:, 2].max() * 255
RGBA[:, -1] = 255
init_pts = Points(init_particles, r=6, c=RGBA)
print(f'Data: {init_mesh_path}\n'
      f'Number of particles: {len(init_particles)}')
del init_particles

init_mesh = Mesh(init_mesh_path)
coords = init_mesh.points()
coords += (0.25, 0.25, 0.04)
coords[:, 0] += 0.06
init_mesh.points(coords)

show([init_pts, init_mesh], __doc__, axes=True).close()
del init_pts, init_mesh

end_mesh_path = os.path.join(script_path, '..', f'data-motion-{tr}', f'eef-{agent}',
                         f'mesh_{data_ind}1_repaired_normalised.obj')
end_particles = generate_particles_from_mesh(end_mesh_path, pos=(0.25, 0.25, 0.04), voxelize_res=res, particle_density=particle_density)
RGBA = np.zeros((len(end_particles), 4))
RGBA[:, 0] = end_particles[:, 2] / end_particles[:, 2].max() * 255
RGBA[:, 1] = end_particles[:, 2] / end_particles[:, 2].max() * 255
RGBA[:, -1] = 255
end_pts = Points(end_particles, r=6, c=RGBA)
print(f'Data: {end_mesh_path}\n'
      f'Number of particles: {len(end_particles)}')
del end_particles

end_mesh = Mesh(end_mesh_path)
coords = end_mesh.points()
coords += (0.25, 0.25, 0.04)
coords[:, 0] += 0.06
end_mesh.points(coords)

show([end_pts, end_mesh], __doc__, axes=True).close()
del end_pts, end_mesh
