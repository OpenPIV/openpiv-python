def piv(im1,im2):
    """
    Simplest PIV run on the pair of images using default settings

    piv(im1,im2) will create a tmp.vec file with the vector filed in pix/dt (dt=1) from 
    two images, im1,im2 provided as full path filenames (TIF is preferable, whatever imageio can read)

    """

    import imageio
    import numpy as np
    import matplotlib.pyplot as plt

    from openpiv import process
    
    frame_a = imageio.imread(im1).as_type(np.int32)
    frame_b = imageio.imread(im2).as_type(np.int32)

    u, v =  process.extended_search_area_piv(frame_a,frame_b)
    x, y = process.get_coordinates( image_size=frame_a.shape, 
                               window_size=24, overlap=12 )

    plt.figure(figsize=(12,12))
    plt.imshow(frame_a,cm=plt.cm.gray)
    plt.quiver(x,y,u,-v)
    plt.show()
    # plt.gca().invert_yaxis()
    
