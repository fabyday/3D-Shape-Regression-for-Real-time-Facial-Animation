import cv2 
import numpy as np
from . import metadata
class Category():
    def __init__(self):
        self.m_category_name = ""


class Image :
    def __init__(self, location, extension : metadata.ImageExtensionEnum, lazy_load : bool = True):
        self.m_image = None 
        self.m_location = location
        self.m_extension = extension
        self.name = ""
        self.m_extension  = metadata.ImageExtension
        self.m_width  = 0 
        self.m_height = 0
        self.m_lazy_load_flag  =lazy_load


    @property
    def image (self):
        if self.m_image is None :
            self.m_image = cv2.imread(self.name)
        return self.m_image
    
    @image.setter
    def image(self, image):
        self.m_image = image


    @property
    def name(self):
        return self.name


    @name.setter
    def name(self, name :str):
        if self.m_lazy_load_flag:
            self.name = name
        else:
            self.m_image = cv2.imread(name)

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
    def load(meta_info : metadata.BaseMeta, lazy_load_flag = True):
        if meta_info.meta_type  == metadata.ImageMeta.META_NAME:
            ImageLoaderFactory.load_from_landmark_meta(meta_info, lazy_load_flag) 

        elif meta_info.meta_type == metadata.LandmarkMeta.META_NAME:
            ImageLoaderFactory.load_from_landmark_meta(meta_info, lazy_load_flag)

    @staticmethod
    def load_from_landmark_meta(meta_info : metadata.ImageMeta, lazy_load_flag : bool):
        ext = meta_info.extension
        location = meta_info.file_location
        for info in meta_info:
            img_obj = Image(location = location, extension=ext, lazy_load=lazy_load_flag)
            img_obj.name = info.m_name

    def load_from_image_meta(meta_info : metadata.LandmarkMeta):
        pass
        


    

