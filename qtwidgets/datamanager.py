

import os 
import os.path as osp

import yaml
import metadata
import image


class BaseMeshMeta():
    """
        provide mapping
    """
    def __init__(self):
        pass 



class ICT_MeshMeta(BaseMeshMeta):
    pass


def load_from_meta():
    pass

class DataCollection:

    class Data():
        def __init__(self):
            self.m_image = None 
            self.m_lmk = None 
        
        

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
        for meta in meta_data:
            data = DataCollection.Data()
            
            pass



    def save(self):
        pass 


    def meta_info_from_data(self, meta_info):
        pass



class Detector():

    def __init__(self):
       self.m_is_loaded = False  


    def load_detector(self, path = None):
        import dlib
        if path == None :
            path = "./shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(path) 
        self.m_is_loaded = True

class DataManager:
    def __init__(self):
        self.m_detector = Detector() 

    def reset(self):
        self.m_data_collection = DataCollection()
        self.m_current_selected_data = None
        self.m_meta = None
        

    def load_detector(self, pth  =None ):
        self.m_detector.load_detector(pth)

    def get_data_collection(self):
        return self.m_data_collection
    
    def load_data_from_meta(self, pth):
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
