import os
import matplotlib.pyplot as plt
import imageio.v3 as iio
import matplotlib as mpl
import numpy as np

script_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_path, '..')
data_path = os.path.join(script_path, '..', 'figures', 'real-robot-long-horizon-motions')
mat = 'soil'

mpl.use('Agg')

fig, axes = plt.subplots(1, 11, figsize=(11 * 2, 2))
plt.subplots_adjust(wspace=0.01)
for i in range(11):
    if i < 9:
        img = iio.imread(os.path.join(data_path, mat, f'0{i+1}.png'))
    else:
        img = iio.imread(os.path.join(data_path, mat, f'{i+1}.png'))
    img_ = np.zeros((img.shape[1], img.shape[1], 3)).astype(np.uint8)
    img_[:img.shape[0], :img.shape[1], :] = img
    for n in range(img.shape[1] - img.shape[0]):
        img_[img.shape[0]+n, :, :] = img[-1, :, :]
    axes[i].imshow(img_)
    axes[i].get_xaxis().set_visible(False)
    axes[i].get_yaxis().set_visible(False)
    axes[i].set_frame_on(False)
plt.savefig(os.path.join(data_path, mat, f'img_combine.pdf'), bbox_inches='tight', pad_inches=0, dpi=300)
plt.close(fig)
