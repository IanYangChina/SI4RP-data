import matplotlib as mpl
import matplotlib.pylab as plt
import os
import numpy as np
mpl.use('Qt5Agg')
import argparse


cmap = 'YlOrBr'
script_path = os.path.dirname(os.path.realpath(__file__))


def main(args):
    motion_name = args['motion_name']
    assert motion_name in ['poking', 'poking-shifting']
    motion_id = args['motion_id']
    assert motion_id in [1, 2]
    motion = f'{motion_name}-{motion_id}'
    long_motion = args['long_motion']

    if long_motion:
        motion = 'long-horizon'
    for res in ['32']:
        for agent in ['rectangle', 'cylinder', 'round']:
            data_path = os.path.join(script_path, '..', '..', 'data',
                                     f'data-motion-{motion}', f'eef-{agent}')
            if args['validation_data']:
                assert not long_motion, 'Long-horizon data is for validation purposes only'
                assert motion_id == 2, 'Validation data is only available for the second motion'
                data_path = os.path.join(script_path, '..', '..', 'data',
                                         f'data-motion-{motion}', f'eef-{agent}', 'validation_data')
            for data_id in ['0', '1']:
                hm_1 = np.load(os.path.join(data_path,
                                            f'target_pcd_height_map-{data_id}-res{res}-vdsize0.001.npy'))
                min_val, max_val = np.amin(hm_1), np.amax(hm_1)

                plt.imshow(hm_1, cmap=cmap, vmin=min_val, vmax=max_val)
                plt.xticks([])
                plt.yticks([])
                plt.show()

    data_path = os.path.join(script_path, '..', '..', 'data', 'other_mats', 'slime', 'long-motion-validation')
    hm_1 = np.load(os.path.join(data_path, f'target_pcd_height_map-0-res32-vdsize0.001.npy'))
    min_val, max_val = np.amin(hm_1), np.amax(hm_1)

    plt.imshow(hm_1, cmap=cmap, vmin=min_val, vmax=max_val)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    data_path = os.path.join(script_path, '..', '..', 'data', 'other_mats', 'soil', 'long-motion-validation')
    hm_1 = np.load(os.path.join(data_path, f'target_pcd_height_map-0-res32-vdsize0.001.npy'))
    min_val, max_val = np.amin(hm_1), np.amax(hm_1)

    plt.imshow(hm_1, cmap=cmap, vmin=min_val, vmax=max_val)
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    description = 'This script is used to examine height map data.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--motion_name', type=str, default='poking', help='Name of the motion: poking or poking-shifting')
    parser.add_argument('--motion_id', type=int, default=1, help='ID of the motion: 1 or 2')
    parser.add_argument('--long_motion', action='store_true', help='Examine long-horizon data')
    parser.add_argument('--valid', dest='validation_data', action='store_true', help='Examine validation data')
    arguments = vars(parser.parse_args())
    main(arguments)
