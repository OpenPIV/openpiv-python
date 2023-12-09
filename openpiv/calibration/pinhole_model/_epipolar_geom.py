import numpy as np


__all__ = [
    "line_dist",
    "point_line_dist"
]


def line_dist(O1, r1, O2, r2):
    """Calculate where two rays intersect.
    
    Using two cameras, calculate the world coordinates where two rays intersect.
    This is done through an analytical solution based on direction vectors (r)
    and camera origins/translations (O). 
    
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
    
    try:
        a = (-r22*B[0] + r1r2*B[1])/(r1r2**2 - r12 * r22)
        b = (-r1r2*B[0] + r12*B[1])/(r1r2**2 - r12 * r22)
    except:
        a, b = 0.0, 0.0
    
    # use the a and b to calculate the minimum distance
    l1, l2 = O1 + a*r1, O2 + b*r2
    dist = sum((l1 - l2)**2)**0.5
    coord = (l1 + l2)*0.5
    
    return dist, coord


def point_line_dist(O,r,P):
    """Calculate the distance from a ray and a world coordinate.
    
    Using the origin/translation and direction vectors of a camera, find the
    minimum distance between a world coordinate and a ray. This is done through 
    an analytical solution based on direction vectors (r) and camera 
    origins/translations (O). 
    
    Parameters
    ----------
    O : np.ndarray
        Three element numpy arrays for camera origin/translation.
    r : np.ndarray
        Three element numpy arrays for direction vectors.
    P : np.ndarray
        A numpy array containg X, Y, and Z world coordinates.
    Returns
    -------
    dist : float
        The minimum dinstance between a ray and a world cordinate.
        
    Notes
    -----
    Function taken from MyPTV; all rights reserved. The direct link to this
    repository is provided below.
    https://github.com/ronshnapp/MyPTV
    
    """
    anum = sum([r[i]*(P[i]-O[i]) for i in range(3)])
    adenum = sum([r[i]*r[i] for i in range(3)])
    
    a = anum / adenum
    
    l = [O[i] + a*r[i] for i in range(3)]
    dist = ((l[0]-P[0])**2 + (l[1]-P[1])**2 + (l[2]-P[2])**2)**0.5
    
    return dist