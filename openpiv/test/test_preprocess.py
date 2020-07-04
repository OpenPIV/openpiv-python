import numpy as np
from openpiv.preprocess import dynamic_masking
from skimage import img_as_float
from skimage.color import rgb2gray, rgba2rgb
from skimage.io import imread
import matplotlib.pyplot as plt
import os


def test_dynamic_masking(display_images=False):
    """ test dynamic_masking """

    # I created an image using skimage.data.binary_blobs:
    # img = erosion(binary_blobs(128,.01))+binary_blobs(128,.8)
    # imsave('moon.png',img)
    # it's a moon on a starry night
    test_directory = os.path.split(os.path.abspath(__file__))[0]
    img = rgb2gray(rgba2rgb(imread(os.path.join(test_directory, 'moon.png'))))
    img1 = dynamic_masking(img_as_float(img), method='intensity')
    assert(np.allclose(img[80:84, 80:84], 0.86908039))  # non-zero image
    assert(np.allclose(img1[80:84, 80:84], 0.0))  # not it's black

    if display_images:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img)
        ax[1].imshow(img1)  # see if the moon has gone
        plt.show()
