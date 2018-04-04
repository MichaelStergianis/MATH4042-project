import numpy as np
import cv2


def gaussian1d(x, mu, sigma):
    g = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) /
                                                    (2 * sigma**2))
    return g


def gaussian(size, mu=[0, 0], sigma=[1.0, 1.0]):
    """Produces a gaussian filter function conforming to the parameters provided"""
    # initialize our kernel to be a square kernel
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    x = np.arange(size).reshape([-1, 1]) - (size // 2)

    g1 = gaussian1d(x, mu[0], sigma[0])
    g2 = gaussian1d(x.T, mu[1], sigma[1])
    k = np.dot(g1, g2)

    # create a function that can be used like our filtering interface
    def kernel(patch):
        return patch * k

    return kernel


def median(patch):
    """Returns the median value of a patch"""
    return np.median(patch)


def improved_median(patch):
    """A median filter that makes decisions about how to compute the return value"""
    ## central pixel is swapped with median
    half_width = patch.shape[0] // 2
    # copy the array
    med_patch = np.array(patch)
    med_patch[half_width, half_width] = np.median(patch)

    ## new average is computed
    avg = patch.mean()

    ## compare each value with new average
    if np.all(med_patch > avg):
        ## if all values are greater than average take median of patch and return
        return np.median(med_patch)

    ## if even one of the values is less than the average leave that pixel alone
    return patch[half_width, half_width]

def bilateral(size, mu=[0, 0], sigma=[1.0, 1.0]):
    """Performs bilateral filtering on the image
    patch - the patch on which to operate """
    half_width = size // 2
    # we use a spatial gaussian for distance from central
    s_kernel = gaussian(size, mu, sigma)

    # create the bilateral kernel
    def kernel(patch):
        # if its a patch that is continuous and 0, return 0
        if np.all(patch == 0):
            return patch[half_width, half_width]
        t = patch.dtype
        patch = np.array(patch, dtype=np.float32)
        std_dev = np.std(patch)
        if std_dev == 0.0:
            std_dev += 1e-5

        # we use a intensity gaussian for disparity with central pixel
        i_kernel = gaussian1d(patch, patch.mean(), std_dev)

        spatial_weight = s_kernel(patch)

        W_p = spatial_weight * i_kernel
        return ((1 / W_p.flatten().sum()) * (W_p * patch)).astype(t)

    return kernel

def convolve(img, kernel, width=5):
    """Convolves img with kernel"""
    # make okay for grayscale and colour
    rows, cols = img.shape[:2]
    img = img.reshape([rows, cols, -1])
    channels = img.shape[2]

    # define patch width and half width
    patch_w = width
    half_w = patch_w // 2

    # pad the image
    img_p = cv2.copyMakeBorder(img, half_w, half_w, half_w, half_w, cv2.BORDER_REPLICATE)
    rows_p, cols_p = img_p.shape[:2]
    img_p = img_p.reshape([rows_p, cols_p, -1])

    # create output
    out = np.empty_like(img)

    for i in range(half_w, rows_p - (2 * half_w)):
        for j in range(half_w, cols_p - (2 * half_w)):
            for c in range(channels):
                il, ih = i-half_w, i+half_w+1
                jl, jh = j-half_w, j+half_w+1
                out[il:ih, jl:jh, c] = kernel(img_p[il:ih, jl:jh, c]).flatten().sum()
    return out
