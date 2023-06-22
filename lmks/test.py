import numpy as np 
import cv2 


t = cv2.imread("gen2.jpg")


with open("gen2.txt", 'r') as fp:
    while True:
        s = fp.readline()
        if not s :
            break 
        tmp = s.split(" ")
        x =  int(float(tmp[0]))
        y =  int(float(tmp[1]))
        cv2.circle(t, [x,y], 10, (1,0,0), 3)






cv2.imshow("tes", t)
cv2.waitKey(0)