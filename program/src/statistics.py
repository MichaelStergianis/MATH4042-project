import numpy as np
import cv2


def rmse(img, ground_truth):
    """Calculates the root mean squared error of an image"""
    m, n, k = ground_truth.shape
    rmse_val = (1 / (m * n * k)) * np.linalg.norm(ground_truth - img)

    return rmse_val


def snr(img, ground_truth):
    """Calculates the signal to noise ratio of an image"""
    snr_val = 20 * np.log10(np.linalg.norm(ground_truth) / np.linalg.norm(img))
    return snr_val


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        'img', type=str, help='The image on which to compute statistics')
    ap.add_argument(
        'ground',
        type=str,
        help='The ground truth image on which to compute statistics')
    ap.add_argument(
        '--snr',
        action='store_true',
        help='Whether or not to compute the snr value')
    ap.add_argument(
        '--rmse',
        action='store_true',
        help='Whether or not to compute the rmse value')

    args = ap.parse_args()

    if args.snr:
        print(snr(args.img, args.ground))

    if args.rmse:
        print(rmse(args.img, args.ground))


if __name__ == '__main__':
    main()
