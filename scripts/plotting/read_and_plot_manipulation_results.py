import os
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams["font.weight"] = "normal"
plt.rcParams.update({'font.size': 30})

script_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_path, '..')
param_set = 1
if param_set == 0:
    m_id = 2
    end_img_id = 93
else:
    assert param_set == 1
    m_id = 4
    end_img_id = 152

long_motion = True
save_path = os.path.join(script_path, '..', 'figures', 'result-figs', 'other_mats')
os.makedirs(save_path, exist_ok=True)

cases = ['fewshot', 'oneshot', 'realoneshot-rectangle', 'realoneshot-round', 'realoneshot-cylinder']
case_names = ['12-mix', '6-mix', '1-rec.', '1-round', '1-cyl.']
agents = ['rectangle', 'round', 'cylinder']
for data_id in [0, 1]:
    fig, axes = plt.subplots(12, 11, figsize=(11*2, 12*2))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    if long_motion:
        axes[0, 6].text(-2800, -500, f'Long motion, object {data_id+1}', horizontalalignment='left', fontsize=40)
    else:
        axes[0, 6].text(-3200, -500, f'Level-{param_set+1} contact complexity', horizontalalignment='left', fontsize=40)
    for n in range(5):
        axes[0, 2+n*2].text(-500, -90, f'{case_names[n]}', horizontalalignment='left', fontsize=35)
    for agent_id in [0, 1, 2]:
        agent = agents[agent_id]
        if long_motion:
            if agent == 'cylinder':
                end_img_id = 377
            elif agent == 'rectangle':
                end_img_id = 624
            else:
                end_img_id = 459
        gt_hm = plt.imread(os.path.join(save_path, f'eef-{agent}', f'target_pcd_height_map-{data_id}-res32-vdsize0.001.png'))[20:-20, 20:-20, :]
        gt_particle = plt.imread(os.path.join(save_path, f'eef-{agent}', f'pcd_vis_{data_id}1.png'))
        axes[agent_id*4+1, 0].imshow(gt_particle)
        axes[agent_id*4+1, 0].text(100, -10, 'Ground\ntruth', horizontalalignment='left')
        axes[agent_id*4+2, 0].imshow(gt_hm)
        if agent_id == 0:
            chartBox = axes[agent_id*4+1, 0].get_position()
            axes[agent_id*4+1, 0].set_position([chartBox.x0-0.07, chartBox.y0+0.02, chartBox.width, chartBox.height])
            chartBox = axes[agent_id*4+2, 0].get_position()
            axes[agent_id*4+2, 0].set_position([chartBox.x0-0.07, chartBox.y0+0.02, chartBox.width, chartBox.height])
        elif agent_id == 1:
            chartBox = axes[agent_id*4+1, 0].get_position()
            axes[agent_id*4+1, 0].set_position([chartBox.x0-0.07, chartBox.y0+0.01, chartBox.width, chartBox.height])
            chartBox = axes[agent_id*4+2, 0].get_position()
            axes[agent_id*4+2, 0].set_position([chartBox.x0-0.07, chartBox.y0+0.01, chartBox.width, chartBox.height])
        else:
            chartBox = axes[agent_id*4+1, 0].get_position()
            axes[agent_id*4+1, 0].set_position([chartBox.x0-0.07, chartBox.y0, chartBox.width, chartBox.height])
            chartBox = axes[agent_id*4+2, 0].get_position()
            axes[agent_id*4+2, 0].set_position([chartBox.x0-0.07, chartBox.y0, chartBox.width, chartBox.height])
        for case_id in range(5):
            case = cases[case_id]
            data_dir = os.path.join(script_path, '..', f'optimisation-{case}-param{param_set}-result-figs')
            loss_types = ['PCD\nCD', 'PRT\nCD', 'PCD\nEMD', 'PRT\nEMD']
            for run_id in [0, 1, 2, 3]:
                loss_type = loss_types[run_id]
                for seed in [0, 1, 2]:
                    seed_dir = os.path.join(data_dir, f'run{run_id}', f'seed{seed}')
                    if os.path.isdir(seed_dir):
                        break
                if long_motion:
                    fig_dir = os.path.join(seed_dir, f'validation_tr_imgs-long_motion-{agent}', f'{data_id}')
                else:
                    fig_dir = os.path.join(seed_dir, f'validation_tr_imgs-motion{m_id}-{agent}', f'{data_id}')
                end_fig = plt.imread(os.path.join(fig_dir, f'img_{end_img_id}.png'))
                heightmap = plt.imread(os.path.join(fig_dir, f'end_heightmap.png'))[20:-20, 20:-20, :]
                axes[run_id+agent_id*4, 1+case_id*2].imshow(end_fig)
                axes[run_id+agent_id*4, 1+case_id*2+1].imshow(heightmap)
                if agent_id == 0:
                    chartBox = axes[run_id+agent_id*4, 1+case_id*2].get_position()
                    axes[run_id+agent_id*4, 1+case_id*2].set_position([chartBox.x0, chartBox.y0+0.02, chartBox.width, chartBox.height])
                    chartBox = axes[run_id+agent_id*4, 1+case_id*2+1].get_position()
                    axes[run_id+agent_id*4, 1+case_id*2+1].set_position([chartBox.x0, chartBox.y0+0.02, chartBox.width, chartBox.height])
                elif agent_id == 1:
                    chartBox = axes[run_id+agent_id*4, 1+case_id*2].get_position()
                    axes[run_id+agent_id*4, 1+case_id*2].set_position([chartBox.x0, chartBox.y0+0.01, chartBox.width, chartBox.height])
                    chartBox = axes[run_id+agent_id*4, 1+case_id*2+1].get_position()
                    axes[run_id+agent_id*4, 1+case_id*2+1].set_position([chartBox.x0, chartBox.y0+0.01, chartBox.width, chartBox.height])
                # if run_id+agent_id*5 == 0:
                #     axes[run_id+agent_id*5, case_id*2].set_title('Particles')
                #     axes[run_id+agent_id*5, case_id*2+1].set_title('Heightmap')
                if case_id == 0:
                    axes[run_id+agent_id*4, 1+case_id*2].set_ylabel(f'{loss_type}', rotation='horizontal', horizontalalignment='left')
                    axes[run_id+agent_id*4, 1+case_id*2].yaxis.set_label_coords(-0.75, .2)

    for i in range(12):
        for j in range(11):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].set_frame_on(False)
    if long_motion:
        plt.savefig(os.path.join(save_path, f'long_motion-visualisation-{data_id}.pdf'), bbox_inches='tight', pad_inches=0, dpi=500)
    else:
        plt.savefig(os.path.join(save_path, f'lv{param_set+1}-visualisation-{data_id}.pdf'), bbox_inches='tight', pad_inches=0, dpi=500)
    # plt.show()
    plt.close()