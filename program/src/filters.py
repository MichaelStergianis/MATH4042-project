import numpy as np

def gaussian1d(x, mu, sigma):
    g = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp( -((x - mu)**2) / (2 * sigma**2) )
    return g

def gaussian(size, mu=np.array([0, 0]), sigma=np.array([1.0, 1.0])):
    """Produces a gaussian filter function conforming to the parameters provided"""
    # initialize our kernel to be a square kernel
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
