from . import image as ImageLib
class Landmark:
    def __init__(self):
        self.m_landmark = None
        self.m_image = None 
        



    @property
    def landmark(self):
        return self.m_landmark
    

    @property
    def image(self): 
        return self.m_image
    @image.setter
    def image(self, image : ImageLib.Image ):
        self.m_image = image
        
    