import numpy as np
import cv2


def rmse(img, ground_truth):
    """Calculates the root mean squared error of an image"""
    m, n, k = ground_truth.shape
    rmse_val = (1 / (m * n * k)) * np.linalg.norm(ground_truth - img)

    return rmse_val


def snr(img, ground_truth):
    """Calculates the signal to noise ratio of an image"""
    snr_val = (np.linalg.norm(ground_truth)
               / (np.linalg.norm(img) - np.linalg.norm(ground_truth)))
    return snr_val


def log_snr(img, ground_truth):
    """Calculates the log signal to noise ratio of an image"""
    snr_val = 20 * np.log10(snr(img, ground_truth))
    return snr_val


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        'pairs', type=str, nargs='+', help='The images on which to compute statistics')
    ap.add_argument(
        '--snr',
        action='store_true',
        help='Whether or not to compute the snr value')
    ap.add_argument(
        '--log-snr',
        action='store_true',
        help='Whether or not to compute the snr value')
    ap.add_argument(
        '--rmse',
        action='store_true',
        help='Whether or not to compute the rmse value')

    args = ap.parse_args()
    if (args.rmse is False) and (args.snr is False) and (args.log_snr is False):
        raise ValueError("Must provide at least one statistic to compute")

    # make sure even number of args
    assert(len(args.pairs) % 2 == 0)
    pairs = np.array(args.pairs).reshape([-1, 2])

    fmt_str = "{0:8s}{1:2.5f}"
    for pair in pairs:
        print("Statistics for {}".format(pair[0]))
        img = cv2.imread(pair[0])
        ground = cv2.imread(pair[1])
        if args.snr:
            print(fmt_str.format("SNR:", snr(img, ground)))

        if args.log_snr:
            print(fmt_str.format("logSNR:", log_snr(img, ground)))

        if args.rmse:
            print(fmt_str.format("RMSE:", rmse(img, ground)))


if __name__ == '__main__':
    main()
