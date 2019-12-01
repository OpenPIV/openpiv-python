def piv():
    """
    Simplest PIV run on the pair of images using default settings

    piv(im1,im2) will create a tmp.vec file with the vector filed in pix/dt (dt=1) from 
    two images, im1,im2 provided as full path filenames (TIF is preferable, whatever imageio can read)

    """



    import imageio
    import numpy as np
    import matplotlib.pyplot as plt

    from openpiv import process
    import pkg_resources as pkg
    import os


    # if im1 is None and im2 is None:
    im1 = pkg.resource_filename('openpiv',os.path.join('examples','test5','frame_a.tif'))
    im2 = pkg.resource_filename('openpiv',os.path.join('examples','test5','frame_b.tif'))


    frame_a = imageio.imread(im1)
    frame_b = imageio.imread(im2)
    
    u, v = process.extended_search_area_piv(frame_a.astype(np.int32),frame_b.astype(np.int32),window_size=32,overlap=16)
    x, y = process.get_coordinates( image_size=frame_a.shape, 
                               window_size=32, overlap=16)

    plt.figure(figsize=(12,12))
    plt.imshow(frame_a,cmap=plt.cm.gray,alpha=0.8,origin='lower')
    plt.quiver(x,y,u,v,scale=50,color='r')
    plt.gca().invert_yaxis()
    plt.show()
    
