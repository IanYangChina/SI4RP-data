import os
import pickle as pkl
import numpy as np
from doma.engine.utils.mesh_ops import get_raw_mesh_path, get_voxelized_mesh_path, voxelize_mesh
from doma.engine.configs.macros import COLOR, RHO, MAT_NAME, NOWHERE
from scipy.spatial.transform import Rotation


class Bodies:
    # Class for fluid / rigid bodies represented by MPM particles
    def __init__(self, dim, particle_density):
        self.bodies = []
        self.body_p_volumes = []
        self.body_p_radii = []
        self.materials = []
        self.colors = []
        self.rhos = []
        self.body_ids = []
        self.dim = dim
        self.particle_density = particle_density

    def __len__(self):
        return len(self.bodies)

    def add_body(self, type, filling='random', **kwargs):
        state = np.random.get_state()
        np.random.seed(0)  # fix seed 0

        assert filling in ['random', 'grid', 'natural'], f'Unsupported filling type: {filling}.'
        if type == 'nowhere':
            self.add_nowhere(**kwargs)
        elif type == 'cube':
            self.add_cube(filling=filling, **kwargs)
        elif type == 'cylinder':
            self.add_cylinder(filling=filling, **kwargs)
        elif type == 'ball':
            self.add_ball(filling=filling, **kwargs)
        elif type == 'mesh':
            self.add_mesh(filling=filling, **kwargs)
        else:
            raise NotImplementedError(f'Unsupported body type: {type}.')

        np.random.set_state(state)

    def compute_n_particles(self, volume):
        return int(round(volume * self.particle_density))

    def compute_n_particles_1D(self, length):
        return int(round(length * np.cbrt(self.particle_density)))

    def _add_body(
            self,
            type,
            particles,
            p_volume,
            material,
            color=None,
            euler=(0.0, 0.0, 0.0),
    ):

        if color is not None:
            body_color = color
        else:
            body_color = COLOR[material]

        body_color = np.tile(body_color, [len(particles), 1])
        body_rho = np.full(len(particles), RHO[material])
        body_material = np.full(len(particles), material)
        body_id = np.full(len(particles), len(self.bodies))

        # rotate
        R = Rotation.from_euler('zyx', np.array(euler)[::-1], degrees=True).as_matrix()
        particles_COM = particles.mean(0)
        particles = (R @ (particles - particles_COM).T).T + particles_COM

        self.colors.append(body_color)
        self.rhos.append(body_rho)
        self.materials.append(body_material)
        self.body_ids.append(body_id)
        self.bodies.append(particles)
        self.body_p_volumes.append(p_volume)
        self.body_p_radii.append(np.cbrt((p_volume * 3) / (4 * np.pi)))

        # print(f'===>  {len(particles):7d} particles of {MAT_NAME[material]:>8} {type:>8} added.')

    def sample_cube(self, lower, upper, filling):
        size = upper - lower
        volume = np.prod(size)
        if filling == 'random':
            n_particles = self.compute_n_particles(volume)
            particles = np.random.uniform(low=lower, high=upper, size=(n_particles, self.dim))
        elif filling == 'grid':
            n_x = self.compute_n_particles_1D(size[0])
            n_y = self.compute_n_particles_1D(size[1])
            n_z = self.compute_n_particles_1D(size[2])
            x = np.linspace(lower[0], upper[0], n_x + 1)
            y = np.linspace(lower[1], upper[1], n_y + 1)
            z = np.linspace(lower[2], upper[2], n_z + 1)
            particles = np.stack(np.meshgrid(x, y, z, indexing='ij'), -1).reshape((-1, 3))
        else:
            raise NotImplementedError(f'Unsupported filling type: {filling}.')

        p_volume = volume / len(particles)

        return particles, p_volume

    def add_nowhere(self, n_particles, **kwargs):
        particles = np.tile(np.array(NOWHERE), (n_particles, 1))
        self._add_body('nowhere', particles, **kwargs)

    def add_cube(self, lower, filling, upper=None, size=None, **kwargs):
        lower = np.array(lower)
        if size is not None:
            upper = lower + np.array(size)
        else:
            upper = np.array(upper)
        assert (upper >= lower).all()

        if filling == 'natural':
            filling = 'grid'  # for cube, natural is the same as grid

        particles, p_volume = self.sample_cube(lower, upper, filling)

        self._add_body('cube', particles, p_volume, **kwargs)

    def add_cylinder(self, center, height, radius, filling, **kwargs):
        radius = np.array(radius)
        center = np.array(center)

        if filling == 'natural':
            n_y = self.compute_n_particles_1D(height)
            n_r = self.compute_n_particles_1D(radius)
            particles = []
            for y_layer in np.linspace(center[1] - height / 2, center[1] + height / 2, n_y + 1):
                for r_layer in np.linspace(0, radius, n_r + 1):
                    n_layer = max(self.compute_n_particles_1D(2 * np.pi * r_layer), 1)
                    rad_layer = np.linspace(0, np.pi * 2, n_layer + 1)[:-1]
                    x_layer = np.cos(rad_layer) * r_layer + center[0]
                    z_layer = np.sin(rad_layer) * r_layer + center[2]
                    particles_layer = np.vstack([x_layer, np.repeat(y_layer, n_layer), z_layer])
                    particles.append(particles_layer)
            particles = np.hstack(particles).T
        else:
            # sample a cube first
            cube_lower = np.array([center[0] - radius, center[1] - height / 2.0, center[2] - radius])
            cube_upper = np.array([center[0] + radius, center[1] + height / 2.0, center[2] + radius])
            particles, p_volume = self.sample_cube(cube_lower, cube_upper, filling)

            # reject out-of-boundary particles
            particles_r = np.linalg.norm(particles[:, [0, 2]] - center[[0, 2]], axis=1)
            particles = particles[particles_r <= radius]

        self._add_body('cylinder', particles, p_volume, **kwargs)

    def add_ball(self, center, radius, filling, **kwargs):
        center = np.array(center)

        if filling == 'natural':
            n_r = self.compute_n_particles_1D(radius)
            particles = []
            for r_sphere in np.linspace(0, radius, n_r + 1):
                n_layers = self.compute_n_particles_1D(r_sphere * np.pi)
                for ver_rad_layer in np.linspace(-np.pi / 2, np.pi / 2, n_layers + 1):
                    y_layer = center[1] + np.sin(ver_rad_layer) * r_sphere
                    r_layer = np.sqrt(max(r_sphere ** 2 - (center[1] - y_layer) ** 2, 0))
                    n_particles_layer = max(self.compute_n_particles_1D(2 * np.pi * r_layer), 1)
                    hor_rad_layer = np.linspace(0, np.pi * 2, n_particles_layer + 1)[:-1]
                    x_layer = np.cos(hor_rad_layer) * r_layer + center[0]
                    z_layer = np.sin(hor_rad_layer) * r_layer + center[2]
                    particles_layer = np.vstack([x_layer, np.repeat(y_layer, n_particles_layer), z_layer])
                    particles.append(particles_layer)
            particles = np.hstack(particles).T
            ball_volume = 4 / 3 * np.pi * radius ** 3
            p_volume = ball_volume / len(particles)
        else:
            # sample a cube first
            cube_lower = center - radius
            cube_upper = center + radius
            particles, p_volume = self.sample_cube(cube_lower, cube_upper, filling)

            # reject out-of-boundary particles
            particles_r = np.linalg.norm(particles - center, axis=1)
            particles = particles[particles_r <= radius]

        self._add_body('ball', particles, p_volume, **kwargs)

    def add_mesh(self, file, filling, save_voxelised=True,
                 pos=(0.5, 0.5, 0.5), scale=(1.0, 1.0, 1.0), voxelize_res=128, **kwargs):
        assert filling != 'natural', 'natural filling not supported for body type: mesh.'

        raw_file_path = get_raw_mesh_path(file)
        voxelized_file_path = get_voxelized_mesh_path(file, voxelize_res)
        voxels = voxelize_mesh(raw_file_path, voxelized_file_path, voxelize_res, save_voxels=save_voxelised)

        # sample a cube first
        scale = np.array(scale)
        pos = np.array(pos)
        cube_lower = pos - scale * 0.5
        cube_upper = pos + scale * 0.5
        particles, p_volume = self.sample_cube(cube_lower, cube_upper, filling)

        # reject out-of-boundary particles
        particles = particles[voxels.is_filled((particles - pos) / scale)]
        self._add_body('mesh', particles, p_volume, **kwargs)

    def get(self):
        if len(self.bodies) == 0:
            return None

        else:
            particles = {
                'x': np.concatenate(self.bodies),
                'mat': np.concatenate(self.materials),
                'color': np.concatenate(self.colors),
                'rho': np.concatenate(self.rhos),
                'body_id': np.concatenate(self.body_ids),
                'bodies': {
                    'n': len(self.bodies),
                    'n_particles': [],
                    'particle_ids': [],
                    'p_volumes': self.body_p_volumes,
                    'p_radii': self.body_p_radii,
                }
            }
            for body_id in range(len(self.bodies)):
                particles['bodies']['n_particles'].append(len(self.bodies[body_id]))
                particles['bodies']['particle_ids'].append(np.sort(np.where(particles['body_id'] == body_id)[0]))

            return particles
