import numpy as np 


def normalize(x):
    mean = np.mean(x, 0)
    std = np.std(x, 0)
    normalized_x = (x - mean)/std
    return normalized_x, mean, std

