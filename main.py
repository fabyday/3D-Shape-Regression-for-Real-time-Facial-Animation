import cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
prev_data = None 
    #https://www.researchgate.net/figure/Kinect-camera-intrinsic-parameters-the-resolution-and-sensor-size-information-from-36_tbl1_321354490
import time 

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    time.sleep(1/60)    
    cv2.imshow("VideoFrame", frame)