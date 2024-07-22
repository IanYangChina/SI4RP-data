import matplotlib as mpl
import matplotlib.pylab as plt
import os
import numpy as np
mpl.use('Qt5Agg')


cmap = 'YlOrBr'
script_path = os.path.dirname(os.path.realpath(__file__))
motion_name = 'poking'
assert motion_name in ['poking', 'poking-shifting']
motion_id = 1
assert motion_id in [1, 2]
motion = f'{motion_name}-{motion_id}'
long_motion = False
if long_motion:
    motion = 'long-horizon'
for res in ['32']:
    for agent in ['rectangle', 'cylinder', 'round']:
        data_path = os.path.join(script_path, '..', '..', 'data', f'data-motion-{motion}', f'eef-{agent}')
        extra_data = False
        if extra_data:
            data_path = os.path.join(script_path, '..', '..', 'data', f'data-motion-{motion}', f'eef-{agent}', 'extra_data')
        validation_data = False
        if validation_data:
            data_path = os.path.join(script_path, '..', '..', 'data', f'data-motion-{motion}', f'eef-{agent}', 'validation_data')
        for data_id in range(2):
            hm_1 = np.load(os.path.join(data_path,
                                        f'target_pcd_height_map-{data_id}-res{res}-vdsize0.001.npy'))
            min_val, max_val = np.amin(hm_1), np.amax(hm_1)

            plt.imshow(hm_1, cmap=cmap, vmin=min_val, vmax=max_val)
            plt.xticks([])
            plt.yticks([])
            plt.show()
