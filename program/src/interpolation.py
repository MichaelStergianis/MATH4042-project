import numpy as np


def interpolate(img, x, y):
    """Perform bi-linear interpolation"""
    w, h = img.shape
    xs = np.array((np.floor(x), np.ceil(x)), dtype=np.int)
    ys = np.array((np.floor(y), np.ceil(y)), dtype=np.int)
    if xs[0] < 0 or xs[1] >= w or ys[0] < 0 or ys[1] >= h:
        return 0

    coef = 1 / (((xs[1] - xs[0]) * (ys[1] - ys[0])) + 0.001)
    X = np.array([[xs[1] - x, x - xs[0]]])
    Y = np.array([[ys[1] - y], [y - ys[0]]])
    Q = np.array([[img[xs[0], ys[0]], img[xs[1], ys[1]]],
                  [img[xs[1], ys[0]], img[xs[1], ys[1]]]])

    ret = (coef * np.matmul(X, np.matmul(Q, Y)))

    return ret[0, 0]
