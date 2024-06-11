# from . import image as ImageLib
import image as ImageLib
import numpy as np 


class Landmark:
    

    class NotInitializedLandmark(Exception):
        pass
    def __init__(self):
        self.m_landmark = None
        self.m_image = None 
        self.m_lmk_meta = None
        

    

    @property
    def landmark(self):
        return self.m_landmark

    def __getitem__(self, args):
        return self.m_landmark[args]
    
    def __setitem__(self, index_tuple, key):
        if self.m_landmark == None :
            raise Landmark.NotInitializedLandmark("landmark was not Initialized")
        self.landmark[index_tuple] = key

    def shape(self):
        return self.m_landmark.shape

    @landmark.setter
    def landmark(self, landmark):
        if isinstance(landmark, np.ndarray):
            self.m_landmark = landmark

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
    lmk[0,0] = 100
    print(lmk.landmark)
    lmk.landmark[0, :] = np.array([[333,333,332]])
    print(lmk.landmark)