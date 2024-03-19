import os
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams["font.weight"] = "bold"
plt.rcParams.update({'font.size': 25})

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

script_path = os.path.dirname(os.path.abspath(__file__))
for case in ['fewshot', 'oneshot', 'realoneshot-cylinder', 'realoneshot-rectangle', 'realoneshot-round']:
    result_path = os.path.join(script_path, '..', f'optimisation-{case}-param1-result-figs')
    for run_id in [0, 1, 2, 3]:
        for seed in [0, 1, 2]:
            seed_path = os.path.join(result_path, f'run{run_id}', f'seed{seed}')
            if not os.path.exists(seed_path):
                continue
            for agent in ['rectangle', 'round', 'cylinder']:
                if agent == 'rectangle':
                    img_ids = [0, 62, 124, 186, 248, 310, 372, 434, 496, 558, 624]
                elif agent == 'round':
                    img_ids = [0, 46, 92, 138, 184, 230, 276, 322, 368, 414, 459]
                else:
                    img_ids = [0, 37, 74, 111, 148, 185, 222, 259, 296, 333, 377]

                for data_ind in [0, 1]:
                    data_path = os.path.join(seed_path, f'validation_tr_imgs-long_motion-{agent}', f'{data_ind}')
                    fig, axes = plt.subplots(1, len(img_ids), figsize=(len(img_ids)*2, 2))
                    plt.subplots_adjust(wspace=0.01)
                    for i in range(len(img_ids)):
                        img_id = img_ids[i]
                        img = plt.imread(os.path.join(data_path, f'img_{img_id}.png'))
                        axes[i].imshow(img)
                        axes[i].set_xlabel(f't{img_id}'.translate(SUB))
                        axes[i].set_xticks([])
                        axes[i].get_yaxis().set_visible(False)
                        axes[i].set_frame_on(False)
                    # plt.tight_layout()
                    plt.savefig(os.path.join(data_path, f'img_combine.pdf'), bbox_inches='tight', pad_inches=0)
                    plt.close(fig)