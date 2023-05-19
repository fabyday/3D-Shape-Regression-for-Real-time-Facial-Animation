from typing import Any
import numpy as np 
import geo_func as geo 
import fmath
class BoundingBox:
    def __init__(self, x,y, width, height) -> None:
        self.x = x 
        self.y = y 
        self.width = width
        self.height = height

        self.centroid_x = x + width/2
        self.centroid_y = y + height/2


class FernCascade:
    def __init__(self, second_level_num, beta = 1000):
        self.second_level_num = second_level_num
        self.beta = beta
        self.omega = 0
        self.primitive_regressors = [Fern() for _ in range(second_level_num)]

        self.cov_fi_fj = None 


    def trian(in_imgs , gt_lmks, bboxs):
        self.omega = len(gt_lmks)
        pass
    


    def pred():
        pass



class Fern:
    
    def __init__(self, beta, F = 5) -> None:
        self.beta = beta
        self.F = F




    def Train(self, in_imgs , gt_lmks, bboxs):
        self.omega = len(gt_lmks)
        for i in range(self.F):
            random_direction = np.random.normal(0, 1, size = (np.shape(gt_lmks[0])))
            



    def _delta_s(self, ground_truths, prev_result):
        constant = 1/( (1 + self.beta/self.omega ) * self.omega )
        sums = np.sum((ground_truths - prev_result), 0)
        return constant*sums
    
    def pred(image):
        pass

    def save_model(fp):
        pass



class ShapeRegressor:
    def __init__(self, lmk_num, candidate_feature = 400 , first_level_num = 10, second_level_num = 500, beta = 0, training_samples_num = 0 ):
        # first_level == T = 10
        # second level == K = 500
        self.first_level_num = first_level_num
        self.second_level_num = second_level_num
        self.lmk_num = lmk_num
        self.candidate_feature = candidate_feature # P
        self.Regressors = [ FernCascade(second_level_num) for _ in range(self.first_level_num)]

        
    def train():
        pass

    def pred():
        pass