import matplotlib.pyplot as plt
import numpy as np
from .helper import masks_to_colorimg
from .simulation import generate_random_data
from PIL import Image

# Generate some random images

def main():
    input_images, target_masks = generate_random_data(572, 572, count=1)
    print(input_images)
    # for x in [input_images, target_masks]:
    #     print(x.shape)
    #     print(x.min(), x.max())

    # # Change channel-order and make 3 channels for matplot
    input_images_rgb = [x.astype(np.uint8) for x in input_images]

    # # Map each channel (i.e. class) to each color
    target_masks_rgb = [masks_to_colorimg(x) for x in target_masks]

    # # Left: Input image (black and white), Right: Target mask (6ch)
    # helper.plot_side_by_side([input_images_rgb, target_masks_rgb])

    # counter = 0
    # for image in input_images_rgb:
    #     im = Image.fromarray(image)
    #     im.save(f"images/image_{counter}.jpg")
    #     counter += 1
    #
    # counter = 0
    # for image in target_masks_rgb:
    #     im = Image.fromarray(image)
    #     im.save(f"masks/mask_{counter}.jpg")
    #     counter += 1

if __name__ == '__main__':
    main()
