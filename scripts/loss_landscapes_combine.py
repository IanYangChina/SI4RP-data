import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']# + plt.rcParams['font.serif']
# plt.rcParams["font.weight"] = "bold"
plt.rcParams.update({'font.size': 40})
DTYPE_NP = np.float32

script_path = os.path.dirname(os.path.realpath(__file__))
save_fig_data_path = os.path.join(script_path, '..', 'loss-landscapes-combined')
os.makedirs(save_fig_data_path, exist_ok=True)

loss_types = ['chamfer_loss_pcd',
              'chamfer_loss_particle',
              'emd_point_distance_loss',
              'emd_particle_distance_loss']
loss_titles = ['\nPCD\nCD ', '\nPRT\nCD ', '\nPCD\nEMD', '\nPRT\nEMD']
case_titles = ['12-\nmix', '6-\nmix', '1-\nrec.', '1-\nround', '1-\ncyl.']
# cases = ['m12-few-shot',
#          'm12-one-shot',
#          'm2-realoneshot-rectangle',
#          'm2-realoneshot-round',
#          'm2-realoneshot-cylinder']
# fig, axes = plt.subplots(8, 5, figsize=(5 * 3, 8 * 3))
# plt.subplots_adjust(wspace=0.05, hspace=0.05)
# for case_id in range(5):
#     xy_param = 'E-nu'
#     E_list = np.arange(1e4, 3e5, 10000).astype(DTYPE_NP)
#     nu_list = np.arange(0.01, 0.49, 0.016).astype(DTYPE_NP)
#     E, nu = np.meshgrid(E_list, nu_list)
#     loss_Enu = np.zeros_like(E)
#     for loss_id in range(4):
#         loss_type = loss_types[loss_id]
#         case = cases[case_id]
#         loss = np.load(os.path.join(script_path, '..', f'loss-landscapes-{case}',
#                                     f'{loss_type}_euclidean_{xy_param}-{4e7}pd.npy'))
#         loss = np.flip(loss, axis=0)
#         chartBox = axes[loss_id, case_id].get_position()
#         axes[loss_id, case_id].set_position([chartBox.x0, chartBox.y0+0.025, chartBox.width, chartBox.height])
#         axes[loss_id, case_id].imshow(loss, cmap='YlGnBu')
#         axes[loss_id, case_id].set_xticks([])
#         axes[loss_id, case_id].set_yticks([])
#         if loss_id == 0:
#             axes[loss_id, case_id].set_title(case_titles[case_id], pad=50)
#             if case_id == 2:
#                 axes[loss_id, case_id].set_xlabel('$\\nu$', labelpad=-200, style='italic')
#         if case_id == 0:
#             axes[loss_id, case_id].set_ylabel(loss_titles[loss_id], rotation='horizontal',
#                                               horizontalalignment='left')
#             axes[loss_id, case_id].yaxis.set_label_coords(-1.05, .2)
#             if loss_id == 2:
#                 axes[loss_id, case_id].text(-9, 0.5, '$E$', style='italic')
#
#     axes[3, 0].text(-42, 75, 'Level 1 contact complexity', rotation=90, fontsize=44)
#
#     xy_param = 'rho-yieldstress'
#     rho_list = np.arange(1000, 1999, 33.3).astype(DTYPE_NP)
#     yield_stress_list = np.arange(1000, 20200, 640).astype(DTYPE_NP)
#     rho, yield_stress = np.meshgrid(rho_list, yield_stress_list)
#     loss_rhoys = np.zeros_like(rho)
#     for loss_id in range(4):
#         loss_type = loss_types[loss_id]
#         case = cases[case_id]
#         loss = np.load(os.path.join(script_path, '..', f'loss-landscapes-{case}',
#                                     f'{loss_type}_euclidean_{xy_param}-{4e7}pd.npy'))
#         loss = np.flip(loss, axis=0)
#         axes[loss_id+4, case_id].imshow(loss, cmap='YlGnBu')
#         axes[loss_id+4, case_id].set_xticks([])
#         axes[loss_id+4, case_id].set_yticks([])
#         if case_id == 2 and loss_id == 0:
#             axes[loss_id+4, case_id].set_xlabel('$\\rho$', labelpad=-200, style='italic')
#         if case_id == 0:
#             axes[loss_id+4, case_id].set_ylabel(loss_titles[loss_id], rotation='horizontal',
#                                                 horizontalalignment='left')
#             axes[loss_id+4, case_id].yaxis.set_label_coords(-1.0, .2)
#             if loss_id == 2:
#                 axes[loss_id+4, case_id].text(-9, 0.5, '$\sigma_y$', style='italic')
#
# plt.savefig(os.path.join(save_fig_data_path, f'loss-lv1.pdf'),
#             bbox_inches='tight', pad_inches=0, dpi=500)
# plt.close(fig)

cases = ['m34-few-shot',
         'm34-one-shot',
         'm4-realoneshot-rectangle',
         'm4-realoneshot-round',
         'm4-realoneshot-cylinder']
fig, axes = plt.subplots(12, 5, figsize=(5 * 3, 12 * 3))
plt.subplots_adjust(wspace=0.05, hspace=0.05)
for case_id in range(5):
    xy_param = 'E-nu'
    for loss_id in range(4):
        loss_type = loss_types[loss_id]
        case = cases[case_id]
        loss = np.load(os.path.join(script_path, '..', f'loss-landscapes-{case}',
                                    f'{loss_type}_euclidean_{xy_param}-{4e7}pd.npy'))
        loss = np.flip(loss, axis=0)
        chartBox = axes[loss_id, case_id].get_position()
        axes[loss_id, case_id].set_position([chartBox.x0, chartBox.y0+0.03, chartBox.width, chartBox.height])
        axes[loss_id, case_id].imshow(loss, cmap='YlGnBu')
        axes[loss_id, case_id].set_xticks([])
        axes[loss_id, case_id].set_yticks([])
        if loss_id == 0:
            # axes[loss_id, case_id].set_title(case_titles[case_id], pad=50)
            if case_id == 2:
                axes[loss_id, case_id].set_xlabel('$\\nu$', labelpad=-200, style='italic')
        if case_id == 0:
            axes[loss_id, case_id].set_ylabel(loss_titles[loss_id], rotation='horizontal',
                                              horizontalalignment='left')
            axes[loss_id, case_id].yaxis.set_label_coords(-1.05, .2)
            if loss_id == 2:
                axes[loss_id, case_id].text(-9, 0.5, '$E$', style='italic')

    axes[5, 0].text(-42, 75, 'Level 2 contact complexity', rotation=90, fontsize=44)

    xy_param = 'rho-yieldstress'
    for loss_id in range(4):
        loss_type = loss_types[loss_id]
        case = cases[case_id]
        loss = np.load(os.path.join(script_path, '..', f'loss-landscapes-{case}',
                                    f'{loss_type}_euclidean_{xy_param}-{4e7}pd.npy'))
        loss = np.flip(loss, axis=0)
        chartBox = axes[loss_id+4, case_id].get_position()
        axes[loss_id+4, case_id].set_position([chartBox.x0, chartBox.y0+0.015, chartBox.width, chartBox.height])
        axes[loss_id+4, case_id].imshow(loss, cmap='YlGnBu')
        axes[loss_id+4, case_id].set_xticks([])
        axes[loss_id+4, case_id].set_yticks([])
        if case_id == 2 and loss_id == 0:
            axes[loss_id+4, case_id].set_xlabel('$\\rho$', labelpad=-200, style='italic')
        if case_id == 0:
            axes[loss_id+4, case_id].set_ylabel(loss_titles[loss_id], rotation='horizontal',
                                                horizontalalignment='left')
            axes[loss_id+4, case_id].yaxis.set_label_coords(-1.0, .2)
            if loss_id == 2:
                axes[loss_id+4, case_id].text(-9, 0.5, '$\sigma_y$', style='italic')

    xy_param = 'mf-gf'
    for loss_id in range(4):
        loss_type = loss_types[loss_id]
        case = cases[case_id]
        loss = np.load(os.path.join(script_path, '..', f'loss-landscapes-{case}',
                                    f'{loss_type}_euclidean_{xy_param}-{4e7}pd.npy'))
        loss = np.flip(loss, axis=0)
        axes[loss_id+8, case_id].imshow(loss, cmap='YlGnBu')
        axes[loss_id+8, case_id].set_xticks([])
        axes[loss_id+8, case_id].set_yticks([])
        if case_id == 2 and loss_id == 0:
            axes[loss_id+8, case_id].set_xlabel('$\mu_m$', labelpad=-200, style='italic')
        if case_id == 0:
            axes[loss_id+8, case_id].set_ylabel(loss_titles[loss_id], rotation='horizontal',
                                                horizontalalignment='left')
            axes[loss_id+8, case_id].yaxis.set_label_coords(-1.0, .2)
            if loss_id == 2:
                axes[loss_id+8, case_id].text(-9, 0.5, '$\mu_t$', style='italic')

# plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(save_fig_data_path, f'loss-lv2.pdf'),
            bbox_inches='tight', pad_inches=0, dpi=500)
plt.close(fig)
