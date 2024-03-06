import numpy as np 

import cv2


def pitch(rad):
    rot = np.identity(4)
    rot[1,1] = np.cos(rad); rot[1,2] = -np.sin(rad)
    rot[2,1] = np.sin(rad); rot[2,2] = np.cos(rad)
    return rot


def yaw(rad):
    rot = np.identity(4)
    rot[0,0] = np.cos(rad); rot[0,2] = np.sin(rad)
    rot[2,0] = -np.sin(rad); rot[2,2] = np.cos(rad)
    return rot


def roll(rad):
    rot = np.identity(4)
    rot[1, 1] = np.cos(rad); rot[1,2] = -np.sin(rad)
    rot[2, 1] = np.sin(rad); rot[2,2] = np.cos(rad)

    return rot 




def grad_pitch(rad):
    rot = np.identity(4)
    rot[1,1] = -np.sin(rad); rot[1,2] = -np.cos(rad)
    rot[2,1] = np.cos(rad); rot[2,2] = -np.sin(rad)
    return rot

def grad_yaw(rad):
    rot = np.identity(4)
    rot[0,0] = -np.sin(rad); rot[0,2] = np.cos(rad)
    rot[2,0] = -np.cos(rad); rot[2,2] = -np.sin(rad)
    return rot

def grad_roll(rad):
    rot = np.identity(4)
    rot[1, 1] = -np.sin(rad); rot[1,2] = -np.cos(rad)
    rot[2, 1] = np.cos(rad); rot[2,2] = -np.sin(rad)
    return rot 







def Euler_PYR(p,y,r):
    return pitch(p)@yaw(y)@roll(r)

def grad_Euler_PYR(p,y,r):
    return (\
            grad_pitch(p)@yaw(y)@roll(r)+\
            pitch(p)@grad_yaw(y)@roll(r)+\
            pitch(p)@yaw(y)@grad_roll(r)\
            )


def projection(shape, bbox):
    shape = np.copy(shape)
    shape[:,0] = (shape[:,0] * bbox.centroid_x/2) + bbox.centroid_x
    shape[:,1] = (shape[:,1] * bbox.centroid_y/2) + bbox.centroid_y

    return shape


def reprojection(shape, bbox):
    shape = np.copy(shape)
    shape[:, 0] = shape[:, 0]*bbox.width/2.0 + bbox.centroid_x
    shape[:, 1] = shape[:, 1]*bbox.width/2.0 + bbox.centroid_y 

    return shape



def similarity_transform(A, B):
    # A, B are landkmark
    # A to B matrix RS
    center_A = np.mean(A, 0)
    center_B = np.mean(B, 0)

    centered_A = A - center_A
    centered_B = B - center_B

    covA, _  = cv2.calcCovarMatrix(centered_A, center_A, cv2.COVAR_COLS)
    covB, _  = cv2.calcCovarMatrix(centered_B, center_B, cv2.COVAR_COLS)
    s1 = np.sqrt(np.linalg.norm(covA))
    s2 = np.sqrt(np.linalg.norm(covB))
    scale = s1 /s2

    centered_A = 1/s1*centered_A
    centered_B = 1/s2*centered_B

    num = 0 
    for a, b in zip(centered_A, centered_B):
        num += a[1]*b[0] - a[0]*b[1] # taste like a determinant
    print(centered_A)
    print(centered_B)
    print((centered_A*centered_B))

    den = np.sum((centered_A*centered_B))

    norm  = np.sqrt(num**2 + den**2)
    sin_theta = num / norm 
    cos_theta = den/norm
    

    rot = np.zeros((2,2))
    rot[0, 0] = cos_theta
    rot[0, 1] = -sin_theta
    rot[1, 0] = sin_theta
    rot[1, 1] = cos_theta


    return rot, scale







def similarity_transform2(A, B):
    """
    A to B scale, rot matrix. least square.
    R @ A  = B
    R = (A.At) = B @ A.T @ inv(A, At)
    """
    return B.T @ A @ np.linalg.inv(A.T@A)