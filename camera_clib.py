import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
test_flag = False
def calibrate(path):
    CHECKERBOARD = (6,8) # 체커보드 행과 열당 내부 코너 수
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []
    imgpoints = [] 
    global test_flag

    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None
    images = glob.glob(path)
    for fname in tqdm(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray,CHECKERBOARD,
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)   
            imgpoints.append(corners2)
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        t = 5
        img = cv2.resize(img, [img.shape[0]//t, img.shape[0]//t])
        cv2.imshow('img',img)
        # cv2.waitKey(0)
    cv2.destroyAllWindows()
    h,w = img.shape[:2] # 480, 640
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(mtx)
    if test_flag:
        return mtx, rvecs, tvecs
    return mtx

if __name__ == "__main__":
    test_flag = True
    mtx, rvects, tvecs = calibrate('./images/checker4/Kakao*.jpg')
    CHECKERBOARD = (6,8) # 체커보드 행과 열당 내부 코너 수
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []
    imgpoints = [] 


    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    img = cv2.imread("./images/checker4/KakaoTalk_20230616_172621664_01.jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray,CHECKERBOARD,
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    rot = np.zeros((3,3))
    rot, _ =cv2.Rodrigues(rvects[0] )
    mat = np.zeros((3,3))
    mat[:3,:3] = rot 
    # mat[:, -1] = tvecs[0].reshape(-1)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)   
    test = mtx@(mat@np.squeeze(objp).T +tvecs[0])
    test/=test[-1, :]
    print("test", test.T)
    print("test2" ,np.squeeze(corners2))
    print("test2" ,objp)
    print("test22222")