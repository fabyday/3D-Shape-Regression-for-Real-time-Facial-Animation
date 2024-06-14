

import os 
import os.path as osp

import yaml
import metadata
import image
import qtthread
import image
import landamrk
class BaseMeshMeta():
    """
        provide mapping
    """
    def __init__(self):
        pass 



class ICT_MeshMeta(BaseMeshMeta):
    pass



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
            img_obj = image.Image(location = location, extension=ext, lazy_load=lazy_load_flag)
            img_obj.name = info.name
            img_obj.category = info.category
            img_obj.load()
    
    @staticmethod
    def load_from_landmark_meta(meta_info : metadata.LandmarkMeta, lazy_load_flag : bool = True):
        jobs_object = qtthread.Jobs()
        
        ext  = meta_info.extension
        image_location = meta_info.image_location
        landmark_location = meta_info.file_location
        for info in meta_info:

            img_obj = image.Image(location = image_location, extension=ext, lazy_load=lazy_load_flag)
            img_obj.name = info.m_name
            img_obj.load()


class DataCollection:

    class Data():
        def __init__(self):
            self.m_image = None 
            self.m_lmk =  landamrk.Landmark()
        
        

    def __init__(self):
        self.m_data_item = {}

    def __getitem__(self, key_o_idx):
        if isinstance(key_o_idx, str):
            pass 
        elif isinstance(key_o_idx, int):
            pass
    

    def __iter__(self):
        pass

    def load_from_meta(self, meta_data : metadata.BaseMeta):
        jobs_object = qtthread.Jobs()

        for meta in meta_data:
            data = DataCollection.Data()
            data.m_image
            data.m_lmk
            meta.name
            self.m_data_item[meta.unique_id] = data

        return jobs_object



    def save(self):
        pass 


    def meta_info_from_data(self, meta_info):
        pass





class Detector():

    def __init__(self):
       self.m_is_loaded = False  
       self.m_path = None 

    def set_path(self, pth):
        self.m_path = pth

    def load_detector(self):
        import dlib
        if self.m_path == None :
            self.m_path = "./shape_predictor_68_face_landmarks.dat"
        self.m_detector = dlib.get_frontal_face_detector()
        self.m_predictor = dlib.shape_predictor(self.m_path) 
        self.m_is_loaded = True

    def detect(self, data_object: DataCollection.Data):
        
        h,w = data_object.m_image.shape
        
        rects = self.m_detector(data_object.m_image.image, 1)
        for i, rect in enumerate(rects):
            l = rect.left()
            t = rect.top()
            b = rect.bottom()
            r = rect.right()
            shape = self.m_predictor(data_object.m_image.image, rect)
            for j in range(68):
                x, y = shape.part(j).x, shape.part(j).y
                data_object.m_lmk.lmk[j, :] = (x, y)
    

class LandmarkDetectJob(qtthread.Runnable):
    def __init__(self, data_object : DataCollection.Data, detector : Detector):
        super().__init__()
        self.data_object = data_object
        self.detector = detector

    def run(self):
        self.detector.detect(self.data_object)


class LoadDetectJob(qtthread.Runnable):
    def __init__(self, object : Detector):
        super().__init__()
        self.object = object



    def run(self):
        self.object.load_detector()



class ImageLoadJob(qtthread.Runnable):
    def __init__(self, image_object):
        super().__init__()
        self.image_object = image_object

    def run(self):
        self.image_object.load()

class DataManager:
    def __init__(self):
        self.m_detector = Detector() 

    def reset(self):
        self.m_data_collection = DataCollection()
        self.m_current_selected_data = None
        self.m_meta = None
        

    def load_detector(self, pth  =None ):
        self.m_detector.set_path(pth)
        return LoadDetectJob(self.m_detector)

    def get_data_collection(self):
        return self.m_data_collection
    
    def load_data_from_meta(self, pth):
        """
            return Runnable
        """
        meta_cls = [metadata.LandmarkMeta, metadata.ImageMeta]
        self.reset()
        for cls in meta_cls:
            meta = cls()
            try :
                meta.open_meta(pth)
                break
            except metadata.BaseMeta.MetaTypeNotCompatibleException:
                pass 
        self.m_meta = meta
        print(meta.meta_type, " is loaded ...")
        self.m_data_collection.load_from_meta(self.m_meta)

    def save_data(self, pth):
        pass
    
    




def detect_landmark(data_mgr : DataManager, data:DataCollection.Data):
    data.m_image
    data_mgr.m_detector 
    data.m_lmk



if __name__ == "__main__":
    manger = DataManager()
    pt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_image")
    manger.load_data_from_meta(pt)
