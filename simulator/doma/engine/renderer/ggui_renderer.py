import numpy as np
import taichi as ti
from time import time
from doma.engine.configs.macros import COLOR, TARGET, FRAME, XAXIS, YAXIS, ZAXIS, DTYPE_TI
from doma.engine.utils.transform import construct_homogeneous_transform_matrix


@ti.data_oriented
class GGUIRenderer:
    def __init__(self,
                 res=(480, 480),
                 pcd_gen_res=100,
                 camera_pos=(0.5, 2.5, 3.5),
                 camera_lookat=(0.5, 0.5, 0.5),
                 camera_euler=(135, 0, 180),
                 camera_fov=30,
                 camera_focal_length=0.035,
                 particle_radius=0.003,
                 lights=None,
                 render_agent=True,
                 render_world_frame=True
                 ):
        if lights is None:
            lights = [{'pos': (0.5, 1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
                      {'pos': (0.5, 1.5, 1.5), 'color': (0.5, 0.5, 0.5)}]
        self.res = res
        self.camera_rays = ti.Vector.field(3, dtype=ti.f32, shape=(res[0], res[1]))
        self.camera_pos = np.asarray(camera_pos).reshape(3)
        self.camera_pos_ti = ti.Vector(self.camera_pos)
        self.camera_lookat = np.asarray(camera_lookat).reshape(3)
        self.camera_euler = np.asarray(camera_euler).reshape(3)
        self.transform_world_to_cam = construct_homogeneous_transform_matrix(self.camera_pos, self.camera_euler)
        self.transform_cam_to_world = np.linalg.inv(self.transform_world_to_cam)
        self.transform_cam_to_world_ti = ti.Vector(self.transform_cam_to_world)
        self.camera_fov = camera_fov
        self.camera_focal_length = camera_focal_length
        self.camera_vec = self.camera_pos - self.camera_lookat

        self.point_cloud_gen_res = pcd_gen_res
        self.height_grid_size_x = 0.24  # mm
        self.height_grid_size_y = 0.24
        self.height_grid_xy_offset = (0.2, 0.2)  # centre point of the height map
        self.height_grid_pixel_size_x = self.height_grid_size_x / self.point_cloud_gen_res
        self.height_grid_pixel_size_y = self.height_grid_size_y / self.point_cloud_gen_res
        self.height_grid = ti.field(dtype=ti.f32, shape=(self.point_cloud_gen_res, self.point_cloud_gen_res), needs_grad=False)

        self.camera_pixel_size = 2 * np.tan(np.radians(self.camera_fov / 2)) * self.camera_focal_length / self.point_cloud_gen_res
        self.point_id = ti.field(dtype=ti.i32,
                                 shape=(self.point_cloud_gen_res, self.point_cloud_gen_res))
        self.nearest_distances_to_camera = ti.field(dtype=ti.f32,
                                                    shape=(self.point_cloud_gen_res, self.point_cloud_gen_res))
        self.point_cloud = ti.Vector.field(3, dtype=ti.f32,
                                           shape=(self.point_cloud_gen_res, self.point_cloud_gen_res))

        self.camera_z_far = 1000.0
        self.camera_z_near = 0.1
        self.camera_init_xz_rad = np.arctan2(self.camera_vec[0], self.camera_vec[2])
        self.lights = []
        self.uninit = True

        for light in lights:
            self.add_light(light['pos'], light['color'])

        self.particle_radius = particle_radius

        self.origin_x = ti.Vector.field(3, dtype=ti.f32, shape=(50,))
        self.origin_y = ti.Vector.field(3, dtype=ti.f32, shape=(50,))
        self.origin_z = ti.Vector.field(3, dtype=ti.f32, shape=(50,))

        self.frames = [ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                       ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                       ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                       ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                       ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                       ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                       ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                       ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                       ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                       ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                       ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                       ti.Vector.field(3, dtype=ti.f32, shape=(200,))]

        self.color_target = None
        self.render_agent = render_agent
        self.render_world_frame = render_world_frame

    def add_light(self, pos, color=(0.5, 0.5, 0.5)):
        light = {
            'pos': pos,
            'color': color
        }
        self.lights.append(light)

    def build(self, sim, particles):
        self.sim = sim
        self.distances_to_camera = ti.field(dtype=ti.f32, shape=(self.sim.n_particles,))
        self.particles_color = ti.Vector.field(4, ti.f32, shape=(len(particles['color']),))
        self.particles_color.from_numpy(particles['color'].astype(np.float32))

        self.detector_array_N = 15
        self.detector_array = ti.Vector.field(3, dtype=ti.f32, shape=(self.detector_array_N,))
        self.detector_h = 72
        self.detector_array.from_numpy(np.array([
            [25, self.detector_h, 85],
            [35, self.detector_h, 85],
            [15, self.detector_h, 85],
            [25, self.detector_h, 75],
            [25, self.detector_h, 95],

            [25, self.detector_h, 42],
            [35, self.detector_h, 42],
            [15, self.detector_h, 42],
            [25, self.detector_h, 32],
            [25, self.detector_h, 52],

            [107, self.detector_h, 65],
            [115, self.detector_h, 65],
            [99, self.detector_h, 65],
            [107, self.detector_h, 45],
            [107, self.detector_h, 85],
        ], dtype=np.float32) / 128.0)

        # World frame
        for i in range(50):
            self.origin_x[i] = ti.Vector([0., 0., 0.]) + i / 50 * ti.Vector([0.1, 0., 0.])
            self.origin_y[i] = ti.Vector([0., 0., 0.]) + i / 50 * ti.Vector([0., 0.1, 0.])
            self.origin_z[i] = ti.Vector([0., 0., 0.]) + i / 50 * ti.Vector([0., 0., 0.1])

        # Scene bounding box
        for i in range(200):
            self.frames[0][i] = ti.Vector([0., 0., 0.]) + i / 200 * ti.Vector([1., 0., 0.])
            self.frames[1][i] = ti.Vector([0., 0., 0.]) + i / 200 * ti.Vector([0., 1., 0.])
            self.frames[2][i] = ti.Vector([0., 0., 0.]) + i / 200 * ti.Vector([0., 0., 1.])
            self.frames[3][i] = ti.Vector([1., 1., 1.]) + i / 200 * ti.Vector([-1., 0., 0.])
            self.frames[4][i] = ti.Vector([1., 1., 1.]) + i / 200 * ti.Vector([0., -1., 0.])
            self.frames[5][i] = ti.Vector([1., 1., 1.]) + i / 200 * ti.Vector([0., 0., -1.])
            self.frames[6][i] = ti.Vector([0., 1., 0.]) + i / 200 * ti.Vector([1., 0., 0.])
            self.frames[7][i] = ti.Vector([0., 1., 0.]) + i / 200 * ti.Vector([0., 0., 1.])
            self.frames[8][i] = ti.Vector([1., 0., 0.]) + i / 200 * ti.Vector([0., 1., 0.])
            self.frames[9][i] = ti.Vector([1., 0., 0.]) + i / 200 * ti.Vector([0., 0., 1.])
            self.frames[10][i] = ti.Vector([0., 0., 1.]) + i / 200 * ti.Vector([1., 0., 0.])
            self.frames[11][i] = ti.Vector([0., 0., 1.]) + i / 200 * ti.Vector([0., 1., 0.])

        # timer
        self.t = time()

    def update_fps(self):
        self.fps = 1.0 / (time() - self.t)
        self.t = time()

    def update_camera(self, t=None, rotate=False):
        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)

        speed = 1e-2
        if self.window.is_pressed(ti.ui.UP):
            camera_dir = np.array(self.camera.curr_lookat - self.camera.curr_position)
            camera_dir[1] = 0
            camera_dir /= np.linalg.norm(camera_dir)
            new_camera_pos = np.array(self.camera.curr_position) + camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) + camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)
        elif self.window.is_pressed(ti.ui.DOWN):
            camera_dir = np.array(self.camera.curr_lookat - self.camera.curr_position)
            camera_dir[1] = 0
            camera_dir /= np.linalg.norm(camera_dir)
            new_camera_pos = np.array(self.camera.curr_position) - camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) - camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)
        elif self.window.is_pressed('u'):
            camera_dir = np.array([0, 1, 0])
            new_camera_pos = np.array(self.camera.curr_position) + camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) + camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)
        elif self.window.is_pressed('i'):
            camera_dir = np.array([0, -1, 0])
            new_camera_pos = np.array(self.camera.curr_position) + camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) + camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)
        elif self.window.is_pressed('f'):
            # rotate
            speed = 7.5e-4
            xz_radius = np.linalg.norm([self.camera_vec[0], self.camera_vec[2]])
            rad = speed * np.pi + self.camera_init_xz_rad
            x = xz_radius * np.sin(rad)
            z = xz_radius * np.cos(rad)
            new_camera_pos = np.array([
                x + self.camera_lookat[0],
                self.camera_pos[1],
                z + self.camera_lookat[2]])
            self.camera.position(*new_camera_pos)

        self.scene.set_camera(self.camera)

    def init_window(self, mode):
        show_window = mode == 'human'
        self.window = ti.ui.Window('Visualisation', self.res, vsync=True, show_window=show_window)
        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color((1, 1, 1))
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.up(0.0, 0.0, 1.0)
        self.camera.position(*self.camera_pos)
        self.camera.lookat(*self.camera_lookat)
        self.camera.fov(self.camera_fov)
        self.camera.projection_mode(ti.ui.ProjectionMode.Perspective)
        self.camera.z_far(self.camera_z_far)
        self.camera.z_near(self.camera_z_near)
        self.scene.set_camera(self.camera)

    def update_bounding_box(self, lower, upper):
        delta = upper - lower
        for i in range(200):
            self.frames[0][i] = ti.Vector([lower[0], lower[1], lower[2]]) + i / 200 * ti.Vector([delta[0], 0., 0.])
            self.frames[1][i] = ti.Vector([lower[0], lower[1], lower[2]]) + i / 200 * ti.Vector([0., delta[1], 0.])
            self.frames[2][i] = ti.Vector([lower[0], lower[1], lower[2]]) + i / 200 * ti.Vector([0., 0., delta[2]])
            self.frames[3][i] = ti.Vector([upper[0], upper[1], upper[2]]) + i / 200 * ti.Vector([-delta[0], 0., 0.])
            self.frames[4][i] = ti.Vector([upper[0], upper[1], upper[2]]) + i / 200 * ti.Vector([0., -delta[1], 0.])
            self.frames[5][i] = ti.Vector([upper[0], upper[1], upper[2]]) + i / 200 * ti.Vector([0., 0., -delta[2]])
            self.frames[6][i] = ti.Vector([lower[0], upper[1], lower[2]]) + i / 200 * ti.Vector([delta[0], 0., 0.])
            self.frames[7][i] = ti.Vector([lower[0], upper[1], lower[2]]) + i / 200 * ti.Vector([0., 0., delta[2]])
            self.frames[8][i] = ti.Vector([upper[0], lower[1], lower[2]]) + i / 200 * ti.Vector([0., delta[1], 0.])
            self.frames[9][i] = ti.Vector([upper[0], lower[1], lower[2]]) + i / 200 * ti.Vector([0., 0., delta[2]])
            self.frames[10][i] = ti.Vector([lower[0], lower[1], upper[2]]) + i / 200 * ti.Vector([delta[0], 0., 0.])
            self.frames[11][i] = ti.Vector([lower[0], lower[1], upper[2]]) + i / 200 * ti.Vector([0., delta[1], 0.])

    @ti.func
    def from_xy_to_uv(self, x: DTYPE_TI, y: DTYPE_TI):
        # this does not need to be differentiable as the loss is connected to the z values of the particles/points
        u = (x - self.height_grid_xy_offset[0]) / self.height_grid_pixel_size_x + self.point_cloud_gen_res / 2
        v = (y - self.height_grid_xy_offset[1]) / self.height_grid_pixel_size_y + self.point_cloud_gen_res / 2
        return ti.floor(u, ti.i32), ti.floor(v, ti.i32)

    @ti.kernel
    def calculate_height_grid(self):
        for n in range(self.sim.n_particles):
            u, v = self.from_xy_to_uv(self.sim.particles_render.x[n][0], self.sim.particles_render.x[n][1])
            z = self.sim.particles_render.x[n][2]
            ti.atomic_max(self.height_grid[u, v], z)

    @ti.kernel
    def find_highest_particles(self):
        for n in range(self.sim.n_particles):
            u, v = self.from_xy_to_uv(self.sim.particles_render.x[n][0], self.sim.particles_render.x[n][1])
            if self.sim.particles_render.x[n][2] >= self.height_grid[u, v]:
                self.point_id[u, v] = n
                self.point_cloud[u, v] = self.sim.particles_render.x[n]

    @ti.func
    def compute_euclidean_distance(self, a, b):
        return ti.sqrt(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2))

    @ti.func
    def from_xyz_to_uv(self, x: DTYPE_TI, y: DTYPE_TI, z: DTYPE_TI):
        u = (x * self.camera_focal_length / z) / self.camera_pixel_size + (self.point_cloud_gen_res / 2)
        v = (y * self.camera_focal_length / z) / self.camera_pixel_size + (self.point_cloud_gen_res / 2)
        return ti.floor(u, ti.i32), ti.floor(v, ti.i32)

    @ti.kernel
    def compute_nearest_particle_distances_to_cam(self):
        for n in range(self.sim.n_particles):
            x_in_cam_frame = self.transform_cam_to_world_ti @ ti.Vector([self.sim.particles_render.x[n][0],
                                                                         self.sim.particles_render.x[n][1],
                                                                         self.sim.particles_render.x[n][2],
                                                                         1.0])
            u, v = self.from_xyz_to_uv(x_in_cam_frame[0], x_in_cam_frame[1], x_in_cam_frame[2])
            d = self.compute_euclidean_distance(self.sim.particles_render.x[n], self.camera_pos_ti)
            ti.atomic_min(self.nearest_distances_to_camera[u, v], d)
            self.distances_to_camera[n] = d

    @ti.kernel
    def find_nearest_particles_to_cam(self):
        for n in range(self.sim.n_particles):
            x_in_cam_frame = self.transform_cam_to_world_ti @ ti.Vector([self.sim.particles_render.x[n][0],
                                                                         self.sim.particles_render.x[n][1],
                                                                         self.sim.particles_render.x[n][2],
                                                                         1.0])
            u, v = self.from_xyz_to_uv(x_in_cam_frame[0], x_in_cam_frame[1], x_in_cam_frame[2])
            if self.distances_to_camera[n] <= self.nearest_distances_to_camera[u, v]:
                self.point_cloud[u, v] = self.sim.particles_render.x[n]

    def render_frame(self, mode='human', t=0):
        if self.uninit:
            self.uninit = False
            self.init_window(mode)

        if mode == 'human':
            try:
                self.update_camera(t)
            except:
                self.window.destroy()
                self.init_window(mode=mode)
                self.update_camera(t)

        # world frame
        if self.render_world_frame:
            self.scene.particles(self.origin_x, color=COLOR[XAXIS], radius=self.particle_radius)
            self.scene.particles(self.origin_y, color=COLOR[YAXIS], radius=self.particle_radius)
            self.scene.particles(self.origin_z, color=COLOR[ZAXIS], radius=self.particle_radius)

        # scene bounding box
        # for i in range(12):
        #     self.scene.particles(self.frames[i], color=COLOR[FRAME], radius=self.particle_radius * 0.5)

        # particles
        if self.sim.has_particles:
            state = self.sim.get_state_render(self.sim.cur_substep_local)
            x_particles = state.x
            radius = state[0].radius
            self.scene.particles(x_particles, per_vertex_color=self.particles_color, radius=radius)
            if mode == 'point_cloud':
                self.point_id.fill(-1)
                self.point_cloud.fill(0)
                # self.nearest_distances_to_camera.fill(50)
                # self.compute_nearest_particle_distances_to_cam()
                # self.find_nearest_particles_to_cam()
                self.height_grid.fill(0)
                self.calculate_height_grid()
                self.find_highest_particles()
                points = self.point_cloud.to_numpy().reshape([-1, 3])
                return points

        # statics
        if len(self.sim.statics) != 0:
            for static in self.sim.statics:
                self.scene.mesh(static.vertices, static.faces, per_vertex_color=static.colors)

        # effectors
        if self.render_agent:
            if self.sim.agent is not None and self.sim.agent.n_effectors != 0:
                for effector in self.sim.agent.effectors:
                    if effector.mesh is not None:
                        self.scene.mesh(effector.mesh.vertices, effector.mesh.faces,
                                        per_vertex_color=effector.mesh.colors)

                    # self.update_bounding_box(effector.boundary.lower[None], effector.boundary.upper[None])
                    # for n in range(12):
                    #     self.scene.particles(self.frames[n], color=COLOR[FRAME], radius=self.particle_radius * 0.5)

        for light in self.lights:
            self.scene.point_light(pos=light['pos'], color=light['color'])

        self.canvas.scene(self.scene)

        # camera gui
        # if True:
        #     self.window.GUI.begin("Camera", 0.05, 0.1, 0.2, 0.15)
        #     self.window.GUI.text(f'pos:    {self.camera.curr_position[0]:.2f}, {self.camera.curr_position[1]:.2f}, {self.camera.curr_position[2]:.2f}')
        #     self.window.GUI.text(f'lookat: {self.camera.curr_lookat[0]:.2f}, {self.camera.curr_lookat[1]:.2f}, {self.camera.curr_lookat[2]:.2f}')
        #     self.window.GUI.end()

        if mode == 'human':
            self.window.show()
            return None

        elif mode == 'rgb_array':
            img = np.rot90(self.window.get_image_buffer_as_numpy())[:, :, :3]
            img = (img * 255).astype(np.uint8)
            self.update_fps()
            # print(f'===> GGUIRenderer: {self.fps:.2f} FPS')
            return img

        elif mode == 'depth_array':
            depth = np.rot90(self.window.get_depth_buffer_as_numpy())
            self.update_fps()
            # print(f'===> GGUIRenderer: {self.fps:.2f} FPS')
            return depth
