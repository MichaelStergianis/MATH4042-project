"""A library of interpolation functions"""
import numpy as np

def bilinear(img, x, y):
    """Perform bi-linear interpolation"""
    w, h = img.shape
    # introduce a small fudge factor for when we end up on a pixel exactly
    if np.floor(x).astype(np.float) == x:
        x += 1e-5
    if np.floor(y).astype(np.float) == y:
        y += 1e-5
    xs = np.array((np.floor(x), np.ceil(x)), dtype=np.int)
    ys = np.array((np.floor(y), np.ceil(y)), dtype=np.int)
    if xs[0] < 0 or xs[1] >= w or ys[0] < 0 or ys[1] >= h:
        return 0

    coef = 1 / (((xs[1] - xs[0]) * (ys[1] - ys[0])) + 1e-6)
    X = np.array([[xs[1] - x, x - xs[0]]])
    Y = np.array([[ys[1] - y], [y - ys[0]]])
    Q = np.array([[img[xs[0], ys[0]], img[xs[1], ys[1]]],
                  [img[xs[1], ys[0]], img[xs[1], ys[1]]]])

    ret = (coef * np.matmul(X, np.matmul(Q, Y)))

    return ret[0, 0]


def interpolate(img, func, out_size):
    """Takes an image as an ndarray, an interpolation function with the interface img, x, y; and a new size for the output size"""
    assert len(img.shape) == 3
    rows, cols, channels = img.shape
    o_c, o_r = out_size

    out = np.empty((o_r, o_c, channels))

    # each pixel location in the original image is determined by the following formula
    # loc = (i / o_w) * width
    for i in range(o_r):
        i_prime = i / float(o_r) * rows
        for j in range(o_c):
            j_prime = j / float(o_c) * cols
            for c in range(channels):
                # bilinear treats an image as though it is grayscale, we can fool it with indexing
                out[i, j, c] = func(img[:, :, c], i_prime, j_prime)

    return out


