from typing import Any
import numpy as np 

class BoundingBox:
    def __init__(self, x,y, width, height) -> None:
        self.x = x 
        self.y = y 
        self.width = width
        self.height = height

        self.centroid_x = x + width/2
        self.centroid_y = y + height/2




class Fern:
    


    def __init__(self) -> None:
        
        pass

    



    def Train(in_imgs , gt_lmks, bboxs):
        
        pass
    

    def pred(image):
        pass

    def save_model(path):
        pass



class ShapeRegressor:
    def __init__(self, first_level_num, lmk_num):
        self.first_level_num = first_level_num
        self.lmk_num = lmk_num


    def train():
        pass

    def pred():
        pass