import numpy as np 




def projection(shape, bbox):
    shape = np.copy(shape)
    shape[:,0] = (shape[:,0] * bbox.centroid_x/2) + bbox.centroid_x
    shape[:,1] = (shape[:,1] * bbox.centroid_y/2) + bbox.centroid_y

    return shape


def reprojection(shape, bbox):
    shape = np.copy(shape)
    shape[:, 0] = shape[:, 0]*bbox.width/2.0 + bbox.centroid
    shape[:, 1] = shape[:, 1]*bbox.width/2.0 + bbox.centroid 

    return shape

