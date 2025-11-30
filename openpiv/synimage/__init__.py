"""Synthetic Image Generator Module for OpenPIV.

This module provides tools to generate synthetic PIV images for testing and validation.
"""

from .synimagegen import (
    continuous_flow_field,
    create_synimage_parameters,
    generate_particle_image,
)

__all__ = [
    'continuous_flow_field',
    'create_synimage_parameters',
    'generate_particle_image',
    'synimagegen',
]


def synimagegen(
    image_size=128,
    dt=0.1,
    x_bound=(0, 1),
    y_bound=(0, 1),
    den=0.008,
    par_diam_mean=15 ** (1.0 / 2),
    bit_depth=8,
):
    """Generate a pair of synthetic PIV images with default parameters.
    
    This is a convenience function that creates synthetic PIV image pairs
    with sensible defaults for testing and validation.
    
    Parameters
    ----------
    image_size : int, optional
        Size of the square image in pixels (default: 128)
    dt : float, optional
        Synthetic time difference between images (default: 0.1)
    x_bound : tuple of floats, optional
        X-axis boundaries (default: (0, 1))
    y_bound : tuple of floats, optional
        Y-axis boundaries (default: (0, 1))
    den : float, optional
        Particle density (default: 0.008)
    par_diam_mean : float, optional
        Mean particle diameter in pixels (default: sqrt(15))
    bit_depth : int, optional
        Bit depth of output images (default: 8)
    
    Returns
    -------
    tuple
        (image_a, image_b) - A tuple of two numpy arrays representing
        the synthetic PIV image pair
    
    Examples
    --------
    >>> from openpiv.tools import synimage
    >>> image_a, image_b = synimage.synimagegen(128)
    >>> print(image_a.shape)
    (128, 128)
    """
    # Convert scalar to tuple if needed
    if isinstance(image_size, int):
        img_size = (image_size, image_size)
    else:
        img_size = image_size
    
    # Create synthetic image parameters
    (
        cff,
        conversion_value,
        x1,
        y1,
        U_par,
        V_par,
        par_diam1,
        par_int1,
        x2,
        y2,
        par_diam2,
        par_int2,
    ) = create_synimage_parameters(
        input_data=None,
        x_bound=x_bound,
        y_bound=y_bound,
        image_size=img_size,
        path=None,
        inter=False,
        den=den,
        per_loss_pairs=2,
        par_diam_mean=par_diam_mean,
        par_diam_std=1.5,
        par_int_std=0.25,
        dt=dt,
    )
    
    # Generate the two images
    image_a = generate_particle_image(
        img_size[1], img_size[0], x1, y1, par_diam1, par_int1, bit_depth
    )
    
    image_b = generate_particle_image(
        img_size[1], img_size[0], x2, y2, par_diam2, par_int2, bit_depth
    )
    
    return image_a, image_b
