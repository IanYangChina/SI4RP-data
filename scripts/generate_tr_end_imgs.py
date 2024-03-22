import os
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams["font.weight"] = "bold"
plt.rcParams.update({'font.size': 15})

script_path = os.path.dirname(os.path.realpath(__file__))
param_set = 1
for agent in ['cylinder', 'rectangle', 'round']:
    for id in [0, 1]:
        if param_set == 0:
            m_id = 2
            end_img_id = 93
        else:
            assert param_set == 1
            m_id = 4
            end_img_id = 152
            if agent == 'cylinder':
                end_img_id = 377
            elif agent == 'rectangle':
                end_img_id = 624
            else:
                end_img_id = 459
        save_path = os.path.join(script_path, '..', 'result-figs', f'data-motion-validation', f'eef-{agent}', f'{id}')
        os.makedirs(save_path, exist_ok=True)
        fig, axes = plt.subplots(4, 10, figsize=(10*2, 4*2))
        plt.subplots_adjust(wspace=0.02, hspace=0.02)
        # gt_hm = plt.imread(os.path.join(save_path, '..', f'target_pcd_height_map-{id}-res32-vdsize0.001.png'))
        # gt_particle = plt.imread(os.path.join(save_path, '..', f'pcd_vis_{id}1.png'))
        # axes[0, 0].imshow(gt_particle)
        # axes[0, 0].set_ylabel('Ground Truth')
        # axes[0, 1].imshow(gt_hm)
        cases = ['fewshot', 'oneshot', 'realoneshot-rectangle', 'realoneshot-round', 'realoneshot-cylinder']
        case_names = ['12 datapoints', '6 datapoints', '1 datapoint (rectangle)', '1 datapoint (round)', '1 datapoint (cylinder)']
        for case_id in range(5):
            case = cases[case_id]
            data_dir = os.path.join(script_path, '..', f'optimisation-{case}-param{param_set}-result-figs')
            loss_types = ['PCD CD', 'PRT CD', 'PCD EMD', 'PRT EMD']
            for run_id in [0, 1, 2, 3]:
                loss_type = loss_types[run_id]
                for seed in [0, 1, 2]:
                    seed_dir = os.path.join(data_dir, f'run{run_id}', f'seed{seed}')
                    if os.path.isdir(seed_dir):
                        break
                fig_dir = os.path.join(seed_dir, f'validation_tr_imgs-long_motion-{agent}', f'{id}')
                end_fig = plt.imread(os.path.join(fig_dir, f'img_{end_img_id}.png'))
                heightmap = plt.imread(os.path.join(fig_dir, f'end_heightmap.png'))
                axes[run_id, case_id*2].imshow(end_fig)
                axes[run_id, case_id*2+1].imshow(heightmap)
                if run_id == 0:
                    axes[run_id, case_id*2].set_title('Particles')
                    axes[run_id, case_id*2+1].set_title('Heightmap')
                if case_id == 0:
                    axes[run_id, case_id*2].set_ylabel(f'{loss_type}')

        for i in range(4):
            for j in range(10):
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                axes[i, j].set_frame_on(False)
        plt.savefig(os.path.join(save_path, 'visualisation.pdf'), bbox_inches='tight', pad_inches=0, dpi=500)
        # plt.show()
        plt.close()