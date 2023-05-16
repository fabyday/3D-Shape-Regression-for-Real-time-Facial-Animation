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


class FernCascade:
    def __init__(self, second_level_num, beta, training_samples_num):
        self.second_level_num = second_level_num
        self.beta = beta
        self.omega = training_samples_num

    def trian():
        pass


    def _delta_s(self, ground_truths, prev_result):
        constant = 1/( (1 + self.beta/self.omega ) * self.omega )
        sums = np.sum((ground_truths - prev_result), 0)
        return constant*sums


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
    def __init__(self, lmk_num, first_level_num = 10, second_level_num = 500 ):
        # first_level == T = 10
        # second level == K = 500
        self.first_level_num = first_level_num
        self.second_level_num = second_level_num
        self.lmk_num = lmk_num

        self.Regressors = [ Fern() for _ in range(self.first_level_num)]


    def train():
        pass

    def pred():
        pass