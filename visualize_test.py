import subprocess_helper as vs 

import numpy as np 
import os 



dt = vs.LandmarkDataTransfer(True)
dt.create()
import subprocess


import logging 
import sys 


landmark = np.arange(2*3).reshape(2,3)
disp = np.arange(5*3).reshape(5,3)
index = np.arange(5)

item = vs.LandmarkDataTransfer.Item.create(None)
item.set_disp(disp)
item.set_landmark(landmark)
item.set_nearest_index(index)
dt.send(item)

print(item.header)


# datt = bytes("test\n", "utf-8")



import time  
a = 0 

itert = 100
ii = 0
import os




while True:
    # datt = {"id" : ii, "data" : datt }
    # dt.send(datt)
    # ii += 1 
    # time.sleep(1.0)
    # print("send")
    pass




