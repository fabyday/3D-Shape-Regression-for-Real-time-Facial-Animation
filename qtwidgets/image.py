import cv2 
import numpy as np
import metadata

import os 
import logger




class Category():
    def __init__(self):
        self.m_category_name = ""


class Image :

    class ImageNotLoadedException(Exception):
        pass 


    def __init__(self, location, extension : metadata.ImageExtensionEnum, lazy_load : bool = True):
        self.m_image = None 
        self.m_location = location
        self.m_extension = extension
        self.m_name = ""
        self.m_width  = 0 
        self.m_height = 0
        self.m_lazy_load_flag  = lazy_load

    def resize_by_ratio(self, ratio):
        img = cv2.resize(img, [int(self.m_width/ratio),int(self.m_height/ratio)])

    def resize(self):

        max_length = max(self.m_height, self.m_width)
        if max_length > 1000:
            max_length_ratio = max_length / 1000
        else :
            max_length_ratio = 1

        img = cv2.resize(self.image, [int(self.m_width/max_length_ratio),int(self.m_height/max_length_ratio)])
        return img, max_length_ratio

    @property
    def image (self):
        return self._load()
    
    @image.setter
    def image(self, image : np.ndarray):
        if len(image.shape) == 3 :
            h, w, c = image.shape

        else :
            h, w  = image.shape 
        self.m_width = w 
        self.m_height = h
        self.m_image = image
    
    @property
    def shape(self):
        return self.m_width, self.m_height
    
    def _load(self):
        if self.m_image is None :
            self.m_image = cv2.imread(os.path.join(self.m_location, self.name + self.m_extension.value))
            # if self.m_image is None :
                # raise Image.ImageNotLoadedException("Image is not loaded.")
            if self.m_image is None : 
                return 
            if len(self.m_image.shape) == 3 :
                h, w, _ = self.m_image.shape
            else :
                h, w  = self.m_image.shape 
            self.m_height = h
            self.m_width = w
            return self.m_image
        else:
            return self.m_image

    def load(self):
        if self.m_lazy_load_flag:
            return 
        else : 
            self._load()



    @property
    def name(self):
        return self.m_name


    @name.setter
    def name(self, name :str):
        self.m_name = name
        

    @property
    def extension(self):
        return self.m_extension

    @extension.setter
    def extension(self, extension : str ):
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
            ImageLoaderFactory.load_from_image_meta(meta_info, lazy_load_flag) 

        elif meta_info.meta_type == metadata.LandmarkMeta.META_NAME:
            ImageLoaderFactory.load_from_landmark_meta(meta_info, lazy_load_flag)

    @staticmethod
    def load_from_image_meta(meta_info : metadata.ImageMeta, lazy_load_flag : bool = True):
        ext = meta_info.extension
        print(ext)
        location = meta_info.file_location
        for info in meta_info:
            img_obj = Image(location = location, extension=ext, lazy_load=lazy_load_flag)
            img_obj.name = info.name
            img_obj.category = info.category
            img_obj.load()
    
    @staticmethod
    def load_from_landmark_meta(meta_info : metadata.LandmarkMeta, lazy_load_flag : bool = True):
        ext  = meta_info.extension
        location = meta_info.image_location
        for info in meta_info:
            img_obj = Image(location = location, extension=ext, lazy_load=lazy_load_flag)
            img_obj.name = info.m_name
            img_obj.load()


        


    
if __name__ == "__main__":
    meta_root = "test_image"

    img_meta = metadata.ImageMeta()
    import os 
    
    print(os.path.join(os.path.dirname(__file__), meta_root))
    img_meta.open_meta(os.path.join(os.path.dirname(__file__), meta_root))
    image_list = ImageLoaderFactory.load(img_meta)
    print("pn")
    print("lmk")
    lmk_meta = metadata.LandmarkMeta()
    print(os.path.join(os.path.dirname(__file__), meta_root, "test_lmk_meta"))
    lmk_meta.open_meta(os.path.join(os.path.dirname(__file__), meta_root, "test_lmk_meta"))

