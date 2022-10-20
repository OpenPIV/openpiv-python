""" Test preprocess """
import os
import numpy as np
from skimage import img_as_float
from skimage.color import rgb2gray, rgba2rgb
from imageio.v3 import imread
import matplotlib.pyplot as plt
from openpiv.preprocess import dynamic_masking, mask_coordinates


test_directory = os.path.split(os.path.abspath(__file__))[0]

def test_dynamic_masking(display_images=True):
    """ test dynamic_masking """

    # I created an image using skimage.data.binary_blobs:
    # img = erosion(binary_blobs(128,.01))+binary_blobs(128,.8)
    # imsave('moon.png',img)
    # it's a moon on a starry night
    img = rgb2gray(rgba2rgb(imread(os.path.join(test_directory, "moon.png"))))
    img1, _ = dynamic_masking(img_as_float(img), method="intensity")
    assert np.allclose(img[80:84, 80:84], 0.86908039)  # non-zero image
    assert np.allclose(img1[80:84, 80:84], 0.0)  # now it's black

    if display_images:
        _, ax = plt.subplots(1, 2)
        ax[0].imshow(img)
        ax[1].imshow(img1)  # see if the moon has gone
        plt.show()


def test_mask_coordinates():
    test_directory = os.path.split(os.path.abspath(__file__))[0]
    img = rgb2gray(rgba2rgb(imread(os.path.join(test_directory, "moon.png"))))
    img1, mask = dynamic_masking(img_as_float(img), method="intensity")
    mask_coords = mask_coordinates(mask, 1.5, 3)
    assert(np.allclose(mask_coords, 
            np.array([[127.,  17.],
                [101.,  16.],
                [ 78.,  22.],
                [ 69.,  28.],
                [ 51.,  48.],
                [ 43.,  70.],
                [ 43.,  90.],
                [ 48., 108.],
                [ 57., 127.]])))  # it has to fail so we remember to make a test
