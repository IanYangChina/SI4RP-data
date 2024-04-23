import os
import imageio
import numpy as np

script_path = os.path.dirname(os.path.realpath(__file__))
for n in [0, 1]:
    with imageio.get_writer(os.path.join(script_path, '..', 'video_frames', f'{n}.gif'), mode='I') as writer:
        for i in range(377):
            if i % 5 != 0:
                continue
            image = imageio.v3.imread(os.path.join(script_path, '..', 'video_frames', f'{n}', f'img_{i}.png'))
            writer.append_data(image)
