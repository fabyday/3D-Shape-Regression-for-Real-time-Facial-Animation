import numpy as np 



def pitch(rad):
    rot = np.identity(4)
    rot[1,1] = np.cos(rad); rot[1,2] = -np.sin(rad)
    rot[2,1] = np.sin(rad); rot[2,2] = np.cos(rad)
    return rot


def yaw(rad):
    rot = np.identity(4)
    rot[0,0] = np.cos(rad); rot[0,2] = np.sin(rad)
    rot[2,0] = -np.sin(rad); rot[2,2] = np.cos(rad)
    return rot;


def roll(rad):
    rot = np.identity(4)
    rot[1, 1] = np.cos(rad); rot[1,2] = -np.sin(rad)
    rot[2, 1] = np.sin(rad); rot[2,2] = np.cos(rad)




def Euler_PYR(p,y,r):
    return pitch(p)@yaw(y)@roll(r)





