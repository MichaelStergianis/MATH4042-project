import numpy as np
import cv2


def rmse(img, ground_truth):
    """Calculates the root mean squared error of an image"""
    m, n, k = ground_truth.shape
    rmse_val = 1/(m*n*k)*np.linalg.norm(ground_truth - img)

    return rmse_val


def snr(img, ground_truth):
    """Calculates the signal to noise ratio of an image"""
    snr_val = 20*np.log10(np.linalg.norm(ground_truth)/np.linalg.norm(img))
    return snr_val


# Leanna's test data
#pep = cv2.imread('/home/leanna/Documents/4 year Winter/Imaging/noisy_peppers.png')
#gt = cv2.imread('/home/leanna/Documents/4 year Winter/Imaging/truth_peppers.png')
#sprint(rmse(pep, gt))
