import json
import h5py
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def build_composite_image(image_path1,
			  image_path2,
			  axis=1,
			  margin=0,
			  background=1):
    '''
    Load two images and returns a composite image.

    Parameters
    ----------
    image_path1: Fullpath to image 1.
    image_path2: Fullpath to image 2.
    margin: Space between images

    Returns
    -------
    (Composite image,
	(vertical_offset1, vertical_offset2),
	(horizontal_offset1, horizontal_offset2))
    '''

    background = int(background != 0)
    if axis != 0 and axis != 1:
        raise RuntimeError('Axis must be 0 (vertical) or 1 (horizontal)')

    im1 = mpimg.imread(image_path1)
    im2 = mpimg.imread(image_path2)

    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape

    if axis == 1:
        composite = np.zeros((max(h1, h2), w1 + w2 + margin, 3),
                             ) + 255 * background
        if h1 > h2:
            voff1, voff2 = 0, (h1 - h2) // 2
        else:
            voff1, voff2 = (h2 - h1) // 2, 0
        hoff1, hoff2 = 0, w1 + margin
    else:
        composite = np.zeros((h1 + h2 + margin, max(w1, w2), 3),
                             ) + 255 * background
        if w1 > w2:
            hoff1, hoff2 = 0, (w1 - w2) // 2
        else:
            hoff1, hoff2 = (w2 - w1) // 2, 0
        voff1, voff2 = 0, h1 + margin

    composite[voff1:voff1 + h1, hoff1:hoff1 + w1, :] = im1
    composite[voff2:voff2 + h2, hoff2:hoff2 + w2, :] = im2

    return (composite, (voff1, voff2), (hoff1, hoff2))

