import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import numpy as np


cmap = 'YlOrBr'
script_path = os.path.dirname(os.path.realpath(__file__))
for res in ['32']:
    for tr in ['validation']:
        for agent in ['rectangle', 'cylinder', 'round']:
            for data_id in range(2):
                hm_1 = np.load(os.path.join(script_path, '..', f'data-motion-{tr}', f'eef-{agent}',
                                            f'target_pcd_height_map-{data_id}-res{res}-vdsize0.001.npy'))
                min_val, max_val = np.amin(hm_1), np.amax(hm_1)

                plt.imshow(hm_1, cmap=cmap, vmin=min_val, vmax=max_val)
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(script_path, '..', f'data-motion-{tr}', f'eef-{agent}',
                                         f'target_pcd_height_map-{data_id}-res{res}-vdsize0.001.png'),
                            bbox_inches='tight', dpi=300)
