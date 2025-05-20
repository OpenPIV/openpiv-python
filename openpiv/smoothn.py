import numpy as np
import numpy.ma as ma
from scipy.fftpack import dct, idct
from scipy.optimize import fmin_l_bfgs_b


def smoothn(
    y,
    nS0=10,
    axis=None,
    smoothOrder=2.0,
    sd=None,
    verbose=False,
    s0=None,
    z0=None,
    isrobust=False,
    W=None,
    s=None,
    MaxIter=100,
    TolZ=1e-3,
    weightstr="bisquare",
):
    """Robust spline smoothing for 1-D to N-D data.

    This function provides a fast, automated and robust discretized smoothing
    spline for data of any dimension. It can handle missing values and supports
    robust smoothing that minimizes the influence of outlying data.

    Parameters
    ----------
    y : array_like
        The data to be smoothed. Can be any N-D noisy array (time series,
        images, 3D data, etc.). Non-finite data (NaN or Inf) are treated
        as missing values.
    nS0 : int, optional
        Number of samples used when automatically determining the smoothing
        parameter. Default is 10.
    axis : int or tuple of ints, optional
        Axis or axes along which the smoothing is performed. If None (default),
        smoothing is performed along all axes.
    smoothOrder : float, optional
        Order of the smoothing. Default is 2.0 (equivalent to cubic spline).
    sd : array_like, optional
        Standard deviation of the data. If provided, it is used to compute
        weights as 1/sd^2.
    verbose : bool, optional
        If True, display progress information. Default is False.
    s0 : float, optional
        Initial value for the smoothing parameter. If None (default), it is
        automatically determined.
    z0 : array_like, optional
        Initial guess for the smoothed data. If None (default), the original
        data is used.
    isrobust : bool, optional
        If True, perform robust smoothing that minimizes the influence of
        outlying data. Default is False.
    W : array_like, optional
        Weighting array of positive values, must have the same size as y.
        A zero weight corresponds to a missing value.
    s : float, optional
        Smoothing parameter. If None (default), it is automatically determined
        using the generalized cross-validation (GCV) method. Larger values
        produce smoother results.
    MaxIter : int, optional
        Maximum number of iterations allowed. Default is 100.
    TolZ : float, optional
        Termination tolerance on Z. Must be between 0 and 1. Default is 1e-3.
    weightstr : str, optional
        Type of weight function for robust smoothing. Options are 'bisquare'
        (default), 'cauchy', or 'talworth'.

    Returns
    -------
    z : ndarray
        The smoothed array.
    s : float
        The smoothing parameter used.
    exitflag : int
        Describes the exit condition:
        1 - Convergence was reached
        0 - Maximum number of iterations was reached
        -1 - DCT/IDCT functions not available
    Wtot : ndarray
        The final weighting array used for the smoothing.

    Notes
    -----
    The function uses the discrete cosine transform (DCT) to efficiently
    compute the smoothing. The smoothing parameter s is determined automatically
    using the generalized cross-validation (GCV) method if not provided.

    For robust smoothing, an iteratively re-weighted process is used to
    minimize the influence of outliers.

    Reference
    ---------
    Garcia D, Robust smoothing of gridded data in one and higher dimensions
    with missing values. Computational Statistics & Data Analysis, 2010.
    """
    is_masked = False

    if type(y) == ma.MaskedArray:  # masked array
        is_masked = True
        mask = y.mask
        y = np.array(y)
        y[mask] = 0.0
        if W != None:
            W = np.array(W)
            W[mask] = 0.0
        if sd != None:
            W = np.array(1.0 / sd ** 2)
            W[mask] = 0.0
            sd = None
        y[mask] = np.nan

    if sd is not None:
        sd_ = np.array(sd)
        mask = sd_ > 0.0
        W = np.zeros_like(sd_)
        W[mask] = 1.0 / sd_[mask] ** 2
        sd = None

    if W is not None:
        W = W / np.max(W)

    sizy = y.shape

    # sort axis
    if axis == None:
        axis = tuple(np.arange(y.ndim))

    noe = y.size  # number of elements
    if noe < 2:
        z = y
        exitflag = 0
        Wtot = 0
        return z, s, exitflag, Wtot
    # ---
    # Smoothness parameter and weights
    # if s != None:
    #  s = []
    if W is None:
        W = np.ones(sizy)

    # if z0 == None:
    #  z0 = y.copy()

    # ---
    # "Weighting function" criterion
    weightstr = weightstr.lower()
    # ---
    # Weights. Zero weights are assigned to not finite values (Inf or NaN),
    # (Inf/NaN values = missing data).
    IsFinite = np.array(np.isfinite(y)).astype(bool)
    nof = IsFinite.sum()  # number of finite elements
    W = W * IsFinite
    if np.any(W < 0):
        raise ValueError("smoothn:NegativeWeights", "Weights must all be >=0")
    else:
        # W = W/np.max(W)
        pass
    # ---
    # Weighted or missing data?
    isweighted = np.any(W != 1)
    # ---
    # Robust smoothing?
    # isrobust
    # ---
    # Automatic smoothing?
    isauto = not s
    # ---
    # DCT and IDCT are required
    # We already imported them at the top of the file
    if 'dct' not in globals() or 'idct' not in globals():
        z = y
        exitflag = -1
        Wtot = 0
        return z, s, exitflag, Wtot

    ## Creation of the Lambda tensor
    # ---
    # Lambda contains the eingenvalues of the difference matrix used in this
    # penalized least squares process.
    axis = tuple(np.array(axis).flatten())
    d = y.ndim
    Lambda = np.zeros(sizy)
    for i in axis:
        # create a 1 x d array (so e.g. [1,1] for a 2D case
        siz0 = np.ones((1, y.ndim), dtype=int)[0]
        siz0[i] = sizy[i]
        # cos(pi*(reshape(1:sizy(i),siz0)-1)/sizy(i)))
        # (arange(1,sizy[i]+1).reshape(siz0) - 1.)/sizy[i]
        Lambda = Lambda + (
            np.cos(np.pi * (np.arange(1, sizy[i] + 1) - 1.0) / sizy[i]).reshape(siz0)
        )
        # else:
        #  Lambda = Lambda + siz0
    Lambda = -2.0 * (len(axis) - Lambda)
    if not isauto:
        Gamma = 1.0 / (1 + (s * abs(Lambda)) ** smoothOrder)

    ## Upper and lower bound for the smoothness parameter
    # The average leverage (h) is by definition in [0 1]. Weak smoothing occurs
    # if h is close to 1, while over-smoothing appears when h is near 0. Upper
    # and lower bounds for h are given to avoid under- or over-smoothing. See
    # equation relating h to the smoothness parameter (Equation #12 in the
    # referenced CSDA paper).
    N = sum(np.array(sizy) != 1)
    # tensor rank of the y-array
    hMin = 1e-6
    hMax = 0.99
    # (h/n)**2 = (1 + a)/( 2 a)
    # a = 1/(2 (h/n)**2 -1)
    # where a = sqrt(1 + 16 s)
    # (a**2 -1)/16
    try:
        sMinBnd = np.sqrt(
            (((1 + np.sqrt(1 + 8 * hMax ** (2.0 / N))) / 4.0 / hMax ** (2.0 / N)) ** 2 - 1)
            / 16.0
        )
        sMaxBnd = np.sqrt(
            (((1 + np.sqrt(1 + 8 * hMin ** (2.0 / N))) / 4.0 / hMin ** (2.0 / N)) ** 2 - 1)
            / 16.0
        )
    except:
        sMinBnd = None
        sMaxBnd = None
    ## Initialize before iterating
    # ---
    Wtot = W
    # --- Initial conditions for z
    if isweighted:
        # --- With weighted/missing data
        # An initial guess is provided to ensure faster convergence. For that
        # purpose, a nearest neighbor interpolation followed by a coarse
        # smoothing are performed.
        # ---
        if z0 != None:  # an initial guess (z0) has been provided
            z = z0
        else:
            z = y  # InitialGuess(y,IsFinite);
            z[~IsFinite] = 0.0
    else:
        z = np.zeros(sizy)
    # ---
    z0 = z
    y[~IsFinite] = 0
    # arbitrary values for missing y-data
    # ---
    tol = 1.0
    RobustIterativeProcess = True
    RobustStep = 1
    nit = 0
    # --- Error on p. Smoothness parameter s = 10^p
    errp = 0.1
    # opt = optimset('TolX',errp);
    # --- Relaxation factor RF: to speedup convergence
    RF = 1 + 0.75 * isweighted
    # ??
    ## Main iterative process
    # ---
    if isauto:
        try:
            xpost = np.array([(0.9 * np.log10(sMinBnd) + np.log10(sMaxBnd) * 0.1)])
        except:
            np.array([100.0])
    else:
        xpost = np.array([np.log10(s)])
    while RobustIterativeProcess:
        # --- "amount" of weights (see the function GCVscore)
        aow = sum(Wtot) / noe
        # 0 < aow <= 1
        # ---
        while tol > TolZ and nit < MaxIter:
            if verbose:
                print("tol", tol, "nit", nit)
            nit = nit + 1
            DCTy = dctND(Wtot * (y - z) + z, f=dct)
            if isauto and not np.remainder(np.log2(nit), 1):
                # ---
                # The generalized cross-validation (GCV) method is used.
                # We seek the smoothing parameter s that minimizes the GCV
                # score i.e. s = Argmin(GCVscore).
                # Because this process is time-consuming, it is performed from
                # time to time (when nit is a power of 2)
                # ---
                # errp in here somewhere

                # xpost,f,d = lbfgsb.fmin_l_bfgs_b(gcv,xpost,fprime=None,factr=10.,\
                #   approx_grad=True,bounds=[(log10(sMinBnd),log10(sMaxBnd))],\
                #   args=(Lambda,aow,DCTy,IsFinite,Wtot,y,nof,noe))

                # if we have no clue what value of s to use, better span the
                # possible range to get a reasonable starting point ...
                # only need to do it once though. nS0 is teh number of samples used
                if not s0:
                    ss = np.arange(nS0) * (1.0 / (nS0 - 1.0)) * (
                        np.log10(sMaxBnd) - np.log10(sMinBnd)
                    ) + np.log10(sMinBnd)
                    g = np.zeros_like(ss)
                    for i, p in enumerate(ss):
                        g[i] = gcv(
                            p,
                            Lambda,
                            aow,
                            DCTy,
                            IsFinite,
                            Wtot,
                            y,
                            nof,
                            noe,
                            smoothOrder,
                        )
                        # print 10**p,g[i]
                    xpost = [ss[g == g.min()]]
                    # print '==============='
                    # print nit,tol,g.min(),xpost[0],s
                    # print '==============='
                else:
                    xpost = [s0]
                xpost, f, d = fmin_l_bfgs_b(
                    gcv,
                    xpost,
                    fprime=None,
                    factr=10.0,
                    approx_grad=True,
                    bounds=[(np.log10(sMinBnd), np.log10(sMaxBnd))],
                    args=(Lambda, aow, DCTy, IsFinite, Wtot, y, nof, noe, smoothOrder),
                )
            s = 10 ** xpost[0]
            # update the value we use for the initial s estimate
            s0 = xpost[0]

            Gamma = 1.0 / (1 + (s * abs(Lambda)) ** smoothOrder)

            z = RF * dctND(Gamma * DCTy, f=idct) + (1 - RF) * z
            # if no weighted/missing data => tol=0 (no iteration)
            tol = isweighted * np.linalg.norm(z0 - z) / np.linalg.norm(z)

            z0 = z
            # re-initialization
        exitflag = nit < MaxIter

        if isrobust:  # -- Robust Smoothing: iteratively re-weighted process
            # --- average leverage
            h = np.sqrt(1 + 16.0 * s)
            h = np.sqrt(1 + h) / np.sqrt(2) / h
            h = h ** N
            # --- take robust weights into account
            Wtot = W * RobustWeights(y - z, IsFinite, h, weightstr)
            # --- re-initialize for another iterative weighted process
            isweighted = True
            tol = 1
            nit = 0
            # ---
            RobustStep = RobustStep + 1
            RobustIterativeProcess = RobustStep < 3
            # 3 robust steps are enough.
        else:
            RobustIterativeProcess = False
            # stop the whole process

    ## Warning messages
    # ---
    if isauto:
        if abs(np.log10(s) - np.log10(sMinBnd)) < errp:
            warning(
                "MATLAB:smoothn:SLowerBound",
                [
                    "s = %.3f " % (s)
                    + ": the lower bound for s "
                    + "has been reached. Put s as an input variable if required."
                ],
            )
        elif abs(np.log10(s) - np.log10(sMaxBnd)) < errp:
            warning(
                "MATLAB:smoothn:SUpperBound",
                [
                    "s = %.3f " % (s)
                    + ": the upper bound for s "
                    + "has been reached. Put s as an input variable if required."
                ],
            )
        # warning('MATLAB:smoothn:MaxIter',\
        #    ['Maximum number of iterations (%d'%(MaxIter) + ') has '\
        #    + 'been exceeded. Increase MaxIter option or decrease TolZ value.'])

    if is_masked:
        z = np.ma.array(z, mask=mask)

    return z, s, exitflag, Wtot


def warning(s1, s2):
    print(s1)
    print(s2[0])


## GCV score
# ---
# function GCVscore = gcv(p)
def gcv(p, Lambda, aow, DCTy, IsFinite, Wtot, y, nof, noe, smoothOrder):
    # Search the smoothing parameter s that minimizes the GCV score
    # ---
    s = 10 ** p
    Gamma = 1.0 / (1 + (s * abs(Lambda)) ** smoothOrder)
    # --- RSS = Residual sum-of-squares
    if np.all(aow > 0.9):  # aow = 1 means that all of the data are equally weighted
        # very much faster: does not require any inverse DCT
        RSS = np.linalg.norm(DCTy * (Gamma - 1.0)) ** 2
    else:
        # take account of the weights to calculate RSS:
        yhat = dctND(Gamma * DCTy, f=idct)
        RSS = np.linalg.norm(np.sqrt(Wtot[IsFinite]) * (y[IsFinite] - yhat[IsFinite])) ** 2
    # ---
    TrH = np.sum(Gamma)
    GCVscore = RSS / float(nof) / (1.0 - TrH / float(noe)) ** 2
    return GCVscore


## Robust weights
# function W = RobustWeights(r,I,h,wstr)
def RobustWeights(r, I, h, wstr):
    # weights for robust smoothing.
    MAD = np.median(abs(r[I] - np.median(r[I])))
    # median absolute deviation
    u = abs(r / (1.4826 * MAD) / np.sqrt(1 - h))
    # studentized residuals
    if wstr == "cauchy":
        c = 2.385
        W = 1.0 / (1 + (u / c) ** 2)
        # Cauchy weights
    elif wstr == "talworth":
        c = 2.795
        W = u < c
        # Talworth weights
    else:
        c = 4.685
        W = (1 - (u / c) ** 2) ** 2.0 * ((u / c) < 1)
        # bisquare weights

    W[np.isnan(W)] = 0
    return W


## Initial Guess with weighted/missing data
# function z = InitialGuess(y,I)
def InitialGuess(y, z0):
    """
    Compute initial guess for the smoothed array.

    Parameters
    ----------
    y : ndarray
        The input array to be smoothed.
    z0 : ndarray or None
        Initial guess for the smoothed array. If None, y is used.

    Returns
    -------
    z : ndarray
        The initial guess for the smoothed array.
    """
    # If z0 is provided and has the right size, use it
    if z0 is not None:
        if z0.shape == y.shape:
            return z0
        else:
            # Wrong size, ignore z0
            pass

    # Otherwise, use y as the initial guess
    if isinstance(y, np.ma.MaskedArray):
        # For masked arrays, preserve the mask
        z = y.copy()
    else:
        z = y.copy()

    return z
    # -- coarse fast smoothing using one-tenth of the DCT coefficients
    # siz = z.shape;
    # z = dct(z,norm='ortho',type=2);
    # for k in np.arange(len(z.shape)):
    #    z[ceil(siz[k]/10)+1:-1] = 0;
    #    ss = tuple(roll(array(siz),1-k))
    #    z = z.reshape(ss)
    #    z = np.roll(z.T,1)
    # z = idct(z,norm='ortho',type=2);


# NB: filter is 2*I - (np.roll(I,-1) + np.roll(I,1))


def dctND(data, f=dct):
    nd = len(data.shape)
    if nd == 1:
        return f(data, norm="ortho", type=2)
    elif nd == 2:
        return f(f(data, norm="ortho", type=2).T, norm="ortho", type=2).T
    elif nd == 3:
        return f(
            f(f(data, norm="ortho", type=2, axis=0), norm="ortho", type=2, axis=1),
            norm="ortho",
            type=2,
            axis=2,
        )


def peaks(n):
    """
    Mimic basic of matlab peaks fn

    Parameters
    ----------
    n : int or array_like
        If int, size of the output array. If array, find peaks in this array.

    Returns
    -------
    z : ndarray or list
        If n is int, returns a 2D array with peaks.
        If n is array, returns indices of peaks in the array.
    """
    # If n is an array, find peaks in it
    if isinstance(n, np.ndarray):
        # Find local maxima
        indices = []
        for i in range(1, len(n)-1):
            if n[i] > n[i-1] and n[i] > n[i+1]:
                indices.append(i)
        return indices

    # Otherwise, generate a 2D peaks function
    xp = np.arange(n)
    x, y = np.meshgrid(xp, xp)
    z = np.zeros_like(x).astype(float)
    for i in range(n // 5):
        x0 = np.random.random() * n
        y0 = np.random.random() * n
        sdx = np.random.random() * n / 4.0
        sdy = sdx
        c = np.random.random() * 2 - 1.0
        f = np.exp(
            -(((x - x0) / sdx) ** 2)
            - ((y - y0) / sdy) ** 2
            - (((x - x0) / sdx)) * ((y - y0) / sdy) * c
        )
        f *= np.random.random()
        z += f
    return z


def test1():
    plt.figure(1)
    plt.clf()
    # 1-D example
    x = linspace(0, 100, 2 ** 8)
    y = cos(x / 10) + (x / 50) ** 2 + randn(size(x)) / 10
    y[[70, 75, 80]] = [5.5, 5, 6]
    z = smoothn(y)[0]
    # Regular smoothing
    zr = smoothn(y, isrobust=True)[0]
    # Robust smoothing
    subplot(121)
    plot(x, y, "r.")
    plot(x, z, "k")
    title("Regular smoothing")
    subplot(122)
    plot(x, y, "r.")
    plot(x, zr, "k")
    title("Robust smoothing")


def test2(axis=None):
    # 2-D example
    plt.figure(2)
    plt.clf()
    xp = arange(0, 1, 0.02)
    [x, y] = meshgrid(xp, xp)
    f = exp(x + y) + sin((x - 2 * y) * 3)
    fn = f + (randn(f.size) * 0.5).reshape(f.shape)
    fs = smoothn(fn, axis=axis)[0]
    subplot(121)
    plt.imshow(fn, interpolation="Nearest")
    # axis square
    subplot(122)
    plt.imshow(fs, interpolation="Nearest")
    # axis square


def test3(axis=None):
    # 2-D example with missing data
    plt.figure(3)
    plt.clf()
    n = 256
    y0 = peaks(n)
    y = (y0 + random(shape(y0)) * 2 - 1.0).flatten()
    I = np.random.permutation(range(n ** 2))
    y[I[1 : n ** 2 * 0.5]] = nan
    # lose 50% of data
    y = y.reshape(y0.shape)
    y[40:90, 140:190] = nan
    # create a hole
    yData = y.copy()
    z0, s, exitflag, Wtot = smoothn(yData, axis=axis)
    # smooth data
    yData = y.copy()
    z, s, exitflag, Wtot = smoothn(yData, isrobust=True, axis=axis)
    # smooth data
    y = yData
    vmin = np.min([np.min(z), np.min(z0), np.min(y), np.min(y0)])
    vmax = np.max([np.max(z), np.max(z0), np.max(y), np.max(y0)])
    subplot(221)
    plt.imshow(y, interpolation="Nearest", vmin=vmin, vmax=vmax)
    title("Noisy corrupt data")
    subplot(222)
    plt.imshow(z0, interpolation="Nearest", vmin=vmin, vmax=vmax)
    title("Recovered data #1")
    subplot(223)
    plt.imshow(z, interpolation="Nearest", vmin=vmin, vmax=vmax)
    title("Recovered data #2")
    subplot(224)
    plt.imshow(y0, interpolation="Nearest", vmin=vmin, vmax=vmax)
    title("... compared with original data")


def test4(i=10, step=0.2, axis=None):
    [x, y, z] = mgrid[-2:2:step, -2:2:step, -2:2:step]
    x = array(x)
    y = array(y)
    z = array(z)
    xslice = [-0.8, 1]
    yslice = 2
    zslice = [-2, 0]
    v0 = x * exp(-(x ** 2) - y ** 2 - z ** 2)
    vn = v0 + randn(x.size).reshape(x.shape) * 0.06
    v = smoothn(vn)[0]
    plt.figure(4)
    plt.clf()
    vmin = np.min([np.min(v[:, :, i]), np.min(v0[:, :, i]), np.min(vn[:, :, i])])
    vmax = np.max([np.max(v[:, :, i]), np.max(v0[:, :, i]), np.max(vn[:, :, i])])
    subplot(221)
    plt.imshow(v0[:, :, i], interpolation="Nearest", vmin=vmin, vmax=vmax)
    title("clean z=%d" % i)
    subplot(223)
    plt.imshow(vn[:, :, i], interpolation="Nearest", vmin=vmin, vmax=vmax)
    title("noisy")
    subplot(224)
    plt.imshow(v[:, :, i], interpolation="Nearest", vmin=vmin, vmax=vmax)
    title("cleaned")


def test5():
    t = linspace(0, 2 * pi, 1000)
    x = 2 * cos(t) * (1 - cos(t)) + randn(size(t)) * 0.1
    y = 2 * sin(t) * (1 - cos(t)) + randn(size(t)) * 0.1
    zx = smoothn(x)[0]
    zy = smoothn(y)[0]
    plt.figure(5)
    plt.clf()
    plt.title("Cardioid")
    plot(x, y, "r.")
    plot(zx, zy, "k")


def test6(noise=0.05, nout=30):
    plt.figure(6)
    plt.clf()
    [x, y] = meshgrid(linspace(0, 1, 24), linspace(0, 1, 24))
    Vx0 = cos(2 * pi * x + pi / 2) * cos(2 * pi * y)
    Vy0 = sin(2 * pi * x + pi / 2) * sin(2 * pi * y)
    Vx = Vx0 + noise * randn(24, 24)
    # adding Gaussian noise
    Vy = Vy0 + noise * randn(24, 24)
    # adding Gaussian noise
    I = np.random.permutation(range(Vx.size))
    Vx = Vx.flatten()
    Vx[I[0:nout]] = (rand(nout, 1) - 0.5) * 5
    # adding outliers
    Vx = Vx.reshape(Vy.shape)
    Vy = Vy.flatten()
    Vy[I[0:nout]] = (rand(nout, 1) - 0.5) * 5
    # adding outliers
    Vy = Vy.reshape(Vx.shape)
    Vsx = smoothn(Vx, isrobust=True)[0]
    Vsy = smoothn(Vy, isrobust=True)[0]
    subplot(131)
    quiver(x, y, Vx, Vy, 2.5)
    title("Noisy")
    subplot(132)
    quiver(x, y, Vsx, Vsy)
    title("Recovered")
    subplot(133)
    quiver(x, y, Vx0, Vy0)
    title("Original")


def sparseSVD(D):
    import scipy.sparse

    try:
        import sparsesvd
    except:
        print("bummer ... better get sparsesvd")
        exit(0)
    Ds = scipy.sparse.csc_matrix(D)
    a = sparsesvd.sparsesvd(Ds, Ds.shape[0])
    return a


def sparseTest(n=1000):
    I = np.identity(n)

    # define a 'traditional' D1 matrix
    # which is a right-side difference
    # and which is *not* symmetric :-(
    D1 = np.matrix(I - np.roll(I, 1))
    # so define a symemtric version
    D1a = D1.T - D1

    U, s, Vh = scipy.linalg.svd(D1a)

    # now, get eigenvectors for D1a
    Ut, eigenvalues, Vt = sparseSVD(D1a)
    Ut = np.matrix(Ut)

    # then, an equivalent 2nd O term would be
    D2a = D1a ** 2

    # show we can recover D1a
    D1a_est = Ut.T * np.diag(eigenvalues) * Ut

    # Now, because D2a (& the target D1a) are symmetric:
    D1a_est = Ut.T * np.diag(eigenvalues ** 0.5) * Ut

    D = 2 * I - (np.roll(I, -1) + np.roll(I, 1))
    a = sparseSVD(-D)
    eigenvalues = np.matrix(a[1])
    Ut = np.matrix(a[0])
    Vt = np.matrix(a[2])
    orig = Ut.T * np.diag(np.array(eigenvalues).flatten()) * Vt

    Feigenvalues = np.diag(np.array(np.c_[eigenvalues, 0]).flatten())
    FUt = np.c_[Ut.T, np.zeros(Ut.shape[1])]
    # confirm: FUt * Feigenvalues * FUt.T ~= D

    # m is a 1st O difference matrix
    # with careful edge conditions
    # such that m.T * m = D2
    # D2 being a 2nd O difference matrix
    m = np.matrix(np.identity(100) - np.roll(np.identity(100), 1))
    m[-1, -1] = 0
    m[0, 0] = 1
    a = sparseSVD(m)
    eigenvalues = np.matrix(a[1])
    Ut = np.matrix(a[0])
    Vt = np.matrix(a[2])
    orig = Ut.T * np.diag(np.array(eigenvalues).flatten()) * Vt
    # Vt* Vt.T = I
    # Ut.T * Ut = I
    # ((Vt.T * (np.diag(np.array(eigenvalues).flatten())**2)) * Vt)
    # we see you get the same as m.T * m by squaring the eigenvalues


# from StackOverflow
# https://stackoverflow.com/questions/17115030/want-to-smooth-a-contour-from-a-masked-array

def smooth(u, mask):
    m = ~mask
    r = u*m  # set all 'masked' points to 0. so they aren't used in the smoothing
    a = 4*r[1:-1,1:-1] + r[2:,1:-1] + r[:-2,1:-1] + r[1:-1,2:] + r[1:-1,:-2]
    b = 4*m[1:-1,1:-1] + m[2:,1:-1] + m[:-2,1:-1] + m[1:-1,2:] + m[1:-1,:-2]  # a divisor that accounts for masked points
    b[b==0] = 1.  # for avoiding divide by 0 error (region is masked so value doesn't matter)
    u[1:-1,1:-1] = a/b

def smooth_masked_array(u):
    """ Use smooth() on the masked array """

    if not isinstance(u, np.ma.MaskedArray):
        raise ValueError("Expected masked array")

    m = u.mask

    # run the data through the smoothing filter a few times
    for i in range(10):
        smooth(u, m)

    return np.ma.array(u, mask=m)  # put together the mask and the data
