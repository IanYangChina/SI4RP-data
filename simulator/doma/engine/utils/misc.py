import os
import torch
import socket
import random
import colorsys
import numpy as np
import subprocess as sp
from doma import pkg_root_dir
from doma.engine.configs.macros import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def get_cfg_path(file):
    return os.path.join(pkg_root_dir, 'envs', 'configs', file)


def get_tgt_path(file):
    return os.path.join(pkg_root_dir, 'assets', 'targets', file)


def eval_str(x):
    if type(x) is str:
        return eval(x)
    else:
        return x


def is_on_server():
    hostname = socket.gethostname()
    if 'matrix' in hostname:
        return True
    # elif 'crv' in hostname:
    #     return True
    else:
        return False


def alpha_to_transparency(color):
    return np.array([color[0], color[1], color[2], 1.0 - color[3]])


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_color(h_low=0.0, h_high=1.0, s=1.0, v=1.0, alpha=1.0):
    color = np.array([0.0, 0.0, 0.0, alpha])
    h_range = h_high - h_low
    color[:3] = colorsys.hsv_to_rgb(np.random.rand() * h_range + h_low, s, v)
    return color


@ti.func
def isnan(x):
    '''
    Determine whether the parameter is a number.

    For each element of i of the result, `isnan` returns True
    if x[i] is posititve or negative floating point NaN (Not a Number)
    and False otherwise.

    :parameter x:
        Specifies the value to test for NaN.

    :return:
        The return value is computed as `not (x >= 0 or x <= 0)`.
    '''
    return not (x >= 0 or x <= 0)


@ti.func
def isinf(x):
    '''
    Determine whether the parameter is positive or negative infinity.

    For each element of i of the result, `isinf` returns True
    if x[i] is posititve or negative floating point infinity and
    False otherwise.

    :parameter x:
        Specifies the value to test for infinity.

    :return:
        The return value is computed as `2 * x == x and x != 0`.
    '''
    return 2 * x == x and x != 0


def plot_loss_landscape(p1, p2, loss, fig_title=None, view='left', min_val=None, max_val=None,
                        x_label=None, y_label=None, z_label=None, colorbar=True, cmap='GnBu',
                        hm=False, show=False, save=True, path=None):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams.update({'font.size': 32})
    if not show:
        mpl.use('Agg')
    if hm:
        if min_val is None:
            min_val = np.min(np.min(loss))
        if max_val is None:
            max_val = np.max(np.max(loss))
        loss = np.flip(loss, axis=0)
        fig, ax = plt.subplots()
        im = ax.imshow(loss, cmap=cmap, vmin=min_val, vmax=max_val)
        if colorbar:
            fig.colorbar(im, ax=ax)
        x_ticks = np.linspace(0, p1.shape[1] - 1, 3).astype(np.int32)
        x_ticks = []
        x_labels = np.round(np.linspace(p1[0, 0], p1[0, -1], 3), 2)

        y_ticks = np.linspace(0, p2.shape[0] - 1, 3).astype(np.int32)
        y_ticks = []
        y_labels = np.round(np.linspace(p2[0, 0], p2[-1, 0], 3), 2).tolist()
        y_labels.reverse()
        ax.set_xticks(x_ticks)
        # ax.set_xticklabels(x_labels)
        ax.set_yticks(y_ticks)
        # ax.set_yticklabels(y_labels)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(label=fig_title)
        if save:
            plt.savefig(path, dpi=500, bbox_inches='tight', pad_inches=0)
        if show:
            plt.show()
        plt.close()
    else:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        surf = ax.plot_surface(p1, p2, loss, cmap=cmap,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.zaxis.set_major_locator(LinearLocator(5))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')
        if view == 'left':
            ax.view_init(elev=25., azim=130)
        else:
            ax.view_init(elev=25., azim=-130)
        ax.tick_params(axis='x', pad=0)
        ax.tick_params(axis='y', pad=0)
        ax.tick_params(axis='z', pad=10)
        ax.set_xlabel(x_label)
        ax.xaxis.labelpad = 5
        ax.set_ylabel(y_label)
        ax.yaxis.labelpad = 5
        ax.set_zlabel(z_label)
        ax.zaxis.labelpad = 22
        ax.set_title(label=fig_title)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5, location=view)
        if save:
            plt.savefig(path, dpi=500, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()


def set_parameters(mpm_env, material_id, e, nu, yield_stress=None, rho=None, ground_friction=None,
                   manipulator_friction=None, container_friction=None, sand_friction_angle=None):
    if yield_stress is not None:
        mpm_env.simulator.system_param[None].yield_stress = yield_stress
    mpm_env.simulator.particle_param[material_id].E = e
    mpm_env.simulator.particle_param[material_id].nu = nu
    mpm_env.simulator.particle_param[material_id].mu_temp = e / (2 * (1 + nu))
    mpm_env.simulator.particle_param[material_id].mu_temp_2 = e / (2 * (1 + nu))
    mpm_env.simulator.particle_param[material_id].lam_temp = e * nu / ((1 + nu) * (1 - 2 * nu))
    mpm_env.simulator.particle_param[material_id].lam_temp_2 = e * nu / ((1 + nu) * (1 - 2 * nu))
    if rho is not None:
        mpm_env.simulator.particle_param[material_id].rho = rho
    if ground_friction is not None:
        mpm_env.simulator.system_param[None].ground_friction = ground_friction
    if manipulator_friction is not None:
        mpm_env.simulator.system_param[None].manipulator_friction = manipulator_friction
    if container_friction is not None:
        mpm_env.simulator.system_param[None].container_friction = container_friction
    if sand_friction_angle is not None:
        mpm_env.simulator.system_param[None].sand_friction_angle = sand_friction_angle


def reset_logging(logging):
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.getLogger())
    for logger in loggers:
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True
