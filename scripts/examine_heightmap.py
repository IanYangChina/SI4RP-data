import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import numpy as np


script_path = os.path.dirname(os.path.realpath(__file__))
hm_1 = np.load(os.path.join(script_path, '..', 'data-motion-1', 'eef-round', 'target_pcd_height_map-0-res32-vdsize0.001.npy')) / 1000
hm_2 = np.load(os.path.join(script_path, '..', 'data-motion-1', 'eef-round', 'target_pcd_height_map-2-res32-vdsize0.001.npy')) / 1000
cmap = 'YlOrBr'

min_val, max_val = np.amin(hm_1), np.amax(hm_1)

fig, axs = plt.subplots(1, 2)
axs[0].set_title('Ground truth')
axs[0].xaxis.set_visible(False)
axs[0].yaxis.set_visible(False)
axs[0].imshow(hm_1, cmap=cmap, vmin=min_val, vmax=max_val)

axs[1].set_title('Simulation result')
axs[1].xaxis.set_visible(False)
axs[1].yaxis.set_visible(False)
axs[1].imshow(hm_2, cmap=cmap, vmin=min_val, vmax=max_val)

plt.show()
