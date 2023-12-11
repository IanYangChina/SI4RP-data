import os
import imageio
import numpy as np

tr = str(3)
script_path = os.path.dirname(os.path.realpath(__file__))
with imageio.get_writer(os.path.join(script_path, '..', 'demo_files', f'{tr}.gif'), mode='I') as writer:
    for i in range(200):
        image_np = np.load(os.path.join(script_path, '..', 'demo_files', f'imgs-{tr}', f'img_{i}.npy'))
        image = imageio.core.util.Image(image_np)
        writer.append_data(image)
