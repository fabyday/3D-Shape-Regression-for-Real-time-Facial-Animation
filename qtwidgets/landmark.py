# from . import image as ImageLib
import image as ImageLib
import numpy as np 


class Landmark:
        
    def __init__(self):
        self.m_landmark = None
        self.m_image = None 
        self.m_lmk_meta = None
        



    @property
    def landmark(self):
        return self.m_landmark

    def __getitem__(self, *args):
        return self.m_landmark[args]

    @landmark.setter
    def landmark(self, landmark):
        if isinstance(landmark, np.ndarray):
            self.m_landmark = Landmark.LandmarkObject(landmark)

    @property
    def image(self): 
        return self.m_image
    
    @image.setter
    def image(self, image : ImageLib.Image ):
        self.m_image = image
    
if __name__ == "__main__":
    lmk = Landmark()
    lmk.landmark = np.arange(9).reshape(3,3)
    print(lmk.landmark)
    lmk[0, 1] = 100
    print(lmk.landmark)
    lmk.landmark[0,-1] = 200
    print(lmk.landmark)