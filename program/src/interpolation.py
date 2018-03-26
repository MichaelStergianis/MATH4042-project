"""A library of interpolation functions"""
import numpy as np

BORDER_REPEAT=0
BORDER_REFLECT=1

def bilinear(img, x, y):
    """Perform bi-linear interpolation"""
    rows, cols, channels = img.shape
    # introduce a small fudge factor for when we end up on a pixel exactly
    if np.floor(x).astype(np.float) == x:
        x += 1e-5
    if np.floor(y).astype(np.float) == y:
        y += 1e-5
    xs = np.array((np.floor(x), np.ceil(x)), dtype=np.int)
    ys = np.array((np.floor(y), np.ceil(y)), dtype=np.int)
    if xs[0] < 0 or xs[1] >= rows or ys[0] < 0 or ys[1] >= cols:
        return np.zeros([channels])

    coef = 1 / (((xs[1] - xs[0]) * (ys[1] - ys[0])) + 1e-5)
    X = np.array([[xs[1] - x, x - xs[0]]])
    Y = np.array([[ys[1] - y], [y - ys[0]]])
    Q = np.array([[img[xs[0], ys[0]], img[xs[0], ys[1]]],
                  [img[xs[1], ys[0]], img[xs[1], ys[1]]]])
    ret = np.empty([channels], dtype=np.float)
    for i in range(channels):
        ret[i] = (coef * np.matmul(X, np.matmul(Q[:, :, i], Y)))

    return ret


def interpolate(img, func, out_size):
    """Takes an image as an ndarray, an interpolation function with the interface img, x, y; and a new size for the output size"""
    rows, cols = img.shape[:2]
    img = img.reshape([rows, cols, -1])
    channels = img.shape[2]
    o_c, o_r = out_size

    out = np.empty((o_r, o_c, channels))

    # each pixel location in the original image is determined by the following formula
    # loc = (i / o_w) * width
    for i in range(o_r):
        i_prime = i / float(o_r) * rows
        for j in range(o_c):
            j_prime = j / float(o_c) * cols
            out[i, j] = func(img[:, :], i_prime, j_prime)

    return out.astype(np.uint8)


def pad(img, left, right, bottom, top, scheme=BORDER_REPEAT):
    """Pads the image for convolutions"""
    # create an output that is the input plus each of the border conditions

    # pad the border using the prescribed scheme
    return


def bilateral(img, kernel, n=5):
    """Performs bilateral filtering on the image
    img    - The image
    kernel - The kernel function (gaussian, median, etc)"""
    rows, cols = img.shape[:2]
    img = img.reshape([rows, cols, -1])
    half_width = n // 2
    img_p = pad(img, half_width, half_width, half_width, half_width)
    out = np.empty_like(img)

    # loop over the image
    channels = img.shape[2]
    for y in range(rows):
        for x in range(cols):
            # do processing
            patch = get_patch(x, y, n)
            out[y, x] = img[y, x]

    return
