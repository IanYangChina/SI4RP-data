import os
import argparse
import numpy as np
from doma.engine.utils.misc import plot_loss_landscape
DTYPE_NP = np.float32


def main(args):
    script_path = os.path.dirname(os.path.realpath(__file__))
    if args['exponential_distance']:
        distance_type = 'exponential'
    else:
        distance_type = 'euclidean'

    loss_types = ['point_distance_sr',
                  'point_distance_rs',
                  'chamfer_loss_pcd',
                  'particle_distance_sr',
                  'particle_distance_rs',
                  'chamfer_loss_particle',
                  'height_map_loss_pcd',
                  'emd_point_distance_loss',
                  'emd_particle_distance_loss'
                  ]
    if args['fewshot']:
        fig_data_path = os.path.join(script_path, '..', 'loss-landscapes-m12-few-shot')
    else:
        fig_data_path = os.path.join(script_path, '..', 'loss-landscapes-m12')

    save_fig_data_path = os.path.join(fig_data_path, 'combined')
    os.makedirs(save_fig_data_path, exist_ok=True)

    p_density_str = str(4e7)

    xy_param = 'rho-yieldstress'
    rho_list = np.arange(1000, 1999, 33.3).astype(DTYPE_NP)
    yield_stress_list = np.arange(1000, 20200, 640).astype(DTYPE_NP)
    rho, yield_stress = np.meshgrid(rho_list, yield_stress_list)
    E_ = 2e5
    nu_ = 0.2

    xy_param = 'E-nu'
    E_list = np.arange(1e4, 3e5, 10000).astype(DTYPE_NP)
    nu_list = np.arange(0.01, 0.49, 0.016).astype(DTYPE_NP)
    E, nu = np.meshgrid(E_list, nu_list)
    yield_stress_ = 6e3
    rho_ = 1300.0
    mf = 0.2
    gf = 2.0

    loss_type = ''
    loss_rhoys = np.zeros_like(rho)
    loss_Enu = np.zeros_like(E)
    for i in [2, 7, 8]:
        loss_type += loss_types[i]
        loss_type += '-'
        loss_rhoys += np.load(os.path.join(fig_data_path,
                                           f'{loss_types[i]}_{distance_type}_rho-yieldstress-{p_density_str}pd.npy'))
        loss_Enu += np.load(os.path.join(fig_data_path,
                                         f'{loss_types[i]}_{distance_type}_E-nu-{p_density_str}pd.npy'))
    loss_type = loss_type[:-1]
    print(loss_type)
    fig_title = (f'{loss_type}\n'
                 f'E = {E_}, nu = {nu_}\n'
                 f'm_friction = {mf}, g_friction = {gf}')
    plot_loss_landscape(rho, yield_stress, loss_rhoys, fig_title=fig_title,
                        x_label='rho', y_label='yield_stress', z_label='Loss',
                        hm=True, show=True, save=True,
                        path=os.path.join(save_fig_data_path,
                                          f"{loss_type}_{distance_type}_landscape_rho-yieldstress-topview-{p_density_str}.pdf"))

    fig_title = (f'{loss_type}\n'
                 f'yield_stress = {yield_stress_}, rho = {rho_}\n'
                 f'm_friction = {mf}, g_friction = {gf}')
    plot_loss_landscape(E, nu, loss_Enu, fig_title=fig_title,
                        x_label='E', y_label='nu', z_label='Loss',
                        hm=True, show=True, save=True,
                        path=os.path.join(save_fig_data_path,
                                          f"{loss_type}_{distance_type}_landscape_E-nu-topview-{p_density_str}.pdf"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fewshot', dest='fewshot', default=False, action='store_true')
    parser.add_argument('--exp_dist', dest='exponential_distance', default=False, action='store_true')
    args = vars(parser.parse_args())
    main(args)
