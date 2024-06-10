import cv2 
import numpy as np
from . import metadata
class Category():
    def __init__(self):
        self.m_category_name = ""


class Image :
    def __init__(self):
        self.m_image = None 
        self.name = ""
        self.m_extension  = metadata.ImageExtension
        self.m_width  = 0 
        self.m_height = 0


    @property
    def image (self):
        return self.m_image
    
    @image.setter
    def image(self, image):
        self.m_image = image


    @property
    def name(self):
        return self.name


    @name.setter
    def name(self, name :str):
        self.name = name

    @property
    def extension(self):
        return self.m_extension

    @extension.setter
    def extension(self, extension : str | metadata.ImageExtension ):
        if isinstance(extension , str):
            self.m_extension = metadata.ImageExtension
        elif isinstance(metadata.ImageExtension):
            self.m_extension = extension
        else:
            raise metadata.NotCompatibleExtension()

    @property
    def category(self):
        return self.category
    
    @category.setter
    def category(self, name):
        self.m_category = name
    


class ImageLoaderFactory():



    @staticmethod
    def load(meta_info):
        pass 


    

