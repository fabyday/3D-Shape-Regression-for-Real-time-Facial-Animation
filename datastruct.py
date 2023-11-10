
import numpy as np 



class Data:
    # meta
    def __init__(self):
        self.contour_index = []  
        self.v = [] 
        self.f = []

    @property
    def contour_index(self):
        return self.contour 
    

    @contour_index.setter
    def contour_index(self, cil):
        if isinstance(cil, list) or isinstance(cil, np.ndarray):
            self.contour_index = cil

    



class OptimizerData:
    """

    optimize_sequence data for shape maching.

    for visualizing
    
    """
    def __init__(self):
        self.list = [[]]



    def resize(self, size):
        pass

    def __getitem__(self, index):
        pass




    






