import numpy as np


__all__ = [
    "line_intersect"
]


def line_intersect(O1, r1, O2, r2):
    """Calculate where two rays intersect.
    
    Using two cameras, calculate the world coordinates where two rays
    intersect. This is done through an analytical solution based on 
    direction vectors (r) and camera origins/translations (O). 
    
    Parameters
    ----------
    O1, O2 : np.ndarray
        Three element numpy arrays for camera origin/translation.
    r1, r2 : np.ndarray
        Three element numpy arrays for direction vectors.
        
    Returns
    -------
    dist : float
        The minimum dinstance between the two rays.
    coord : np.ndarray
        The world coordinate that is nearest to the two rays intersecting.
    
    Notes
    -----
    Function taken from MyPTV; all rights reserved. The direct link to this
    repository is provided below.
    https://github.com/ronshnapp/MyPTV
    
    """
    r1r2 = r1[0]*r2[0] + r1[1]*r2[1] + r1[2]*r2[2]
    r12 = r1[0]**2 + r1[1]**2 + r1[2]**2
    r22 = r2[0]**2 + r2[1]**2 + r2[2]**2
    
    dO = O2-O1
    B = [r1[0]*dO[0] + r1[1]*dO[1] + r1[2]*dO[2],
         r2[0]*dO[0] + r2[1]*dO[1] + r2[2]*dO[2]]
    
    # invert matrix to get coefficients a and b
    try:
        a = (-r22*B[0] + r1r2*B[1])/(r1r2**2 - r12 * r22)
        b = (-r1r2*B[0] + r12*B[1])/(r1r2**2 - r12 * r22)
    except:
        a, b = 0.0, 0.0
    
    # now use a and b to calculate the minimum distance
    l1, l2 = O1 + a*r1, O2 + b*r2 # lines
    dist = sum((l1 - l2)**2)**0.5 # minimum distance
    coord = (l1 + l2)*0.5 # coordinate location
    
    return dist, coord