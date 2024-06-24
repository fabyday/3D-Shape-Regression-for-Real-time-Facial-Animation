

import os 
import os.path as osp

import yaml
import metadata
import image
import qtthread
import image
import landmark
import uuid 
import numpy as np 
from typing import Union
import signals

def save_landmark(lmk, path):
    root_path = osp.dirname(path)
    # print("root", root_path)
    if not osp.exists(root_path):
        os.makedirs(root_path)
    with open(path, 'w') as fp:
        lmk_len = len(lmk)
        for i, coord in enumerate(lmk):
            fp.write(str(coord[0])+" "+str(coord[1]))
            if not (lmk_len - 1 == i):
                fp.write("\n")

def load_landmark(path):
    if not osp.exists(path):
        raise FileNotFoundError("File not found. : " + path)
    else : 
        with open(path, 'r') as fp :
            res = []
            while True:
                l = fp.read()
                if len(l) == 0 :
                    break 
                x, y = map(float, l.split(" "))
                res.append([x,y])
            return np.array(res)



class BaseMeshMeta():
    """
        provide mapping
    """
    def __init__(self):
        pass 



class ICT_MeshMeta(BaseMeshMeta):
    pass


class DataCollection:
    class NoDataException(Exception):
        pass 
    
    class Data():
        def __init__(self, uuid):
            self.m_uuid = uuid
            self.m_image = None 
            self.m_lmk =  landmark.Landmark()

        @property
        def unique_id(self):
            return self.m_uuid
        
        

    def __init__(self):
        self.m_data_item = {}

    def __getitem__(self, key_o_idx):
        if len(self.m_data_item) == 0:
            raise DataCollection.NoDataException("data might not be loaded.")

        if isinstance(key_o_idx, (uuid.UUID, str)):
            item = self.m_data_item[key_o_idx]
        elif isinstance(key_o_idx, int):
            key, item = list(self.m_data_item.items())[key_o_idx]
        return item    
    
    def __iter__(self):
        pass
    
    
    def load_from_image_item_meta(self, meta : metadata.ImageMeta, item_meta : metadata.ImageMeta.ImageItemMeta):
        data = DataCollection.Data(item_meta.unique_id)

        ext = meta.extension
        location = meta.file_location

        data.m_image = image.Image(location = location, extension=ext, lazy_load=False)
        data.m_image.name = item_meta.name
        data.m_image.category = item_meta.category
        data.m_lmk = landmark.Landmark()
        self.m_data_item[item_meta.unique_id ] = data
        

    def load_from_lmk_item_meta(self, meta : metadata.LandmarkMeta, item_meta : metadata.LandmarkMeta.LandmarkItemMeta):
        data = DataCollection.Data(item_meta.unique_id)
        
        ext  = meta.extension
        image_location = meta.image_location
        data.m_image = image.Image(location = image_location, extension=ext, lazy_load=False)
        
        data.m_image.name = item_meta.name
        data.m_image.category = item_meta.category
        data.m_image.load()
        data.m_lmk = landmark.Landmark()
        try :
            data.m_lmk.landmark = load_landmark(item_meta.landmark)
        except : 
            pass 

        self.m_data_item[item_meta.unique_id ] = data

    def load_from_meta(self, meta : metadata.BaseMeta, item_meta : metadata.BaseItemMeta):
        if isinstance(item_meta, metadata.LandmarkMeta.LandmarkItemMeta):
            self.load_from_lmk_item_meta(meta, item_meta)

        elif isinstance(item_meta, metadata.ImageMeta.ImageItemMeta):
            self.load_from_image_item_meta(meta, item_meta)



class DataIOFactory():
    @staticmethod
    def load(collection : DataCollection, meta_info : metadata.BaseMeta, lazy_load_flag = True):
        if isinstance(meta_info, metadata.ImageMeta):
            return DataIOFactory.load_from_image_meta(collection, meta_info, lazy_load_flag) 

        elif isinstance(meta_info,metadata.LandmarkMeta):
            return DataIOFactory.load_from_landmark_meta(collection, meta_info, lazy_load_flag)

    @staticmethod
    def load_from_image_meta(collection : DataCollection, meta_info : metadata.ImageMeta, lazy_load_flag : bool = True):
        ext = meta_info.extension
        jobs_object = qtthread.Jobs()
        def run():
            collection.load_from_meta(meta_info, info)
        location = meta_info.file_location
        for info in meta_info.get_item_iterator():
            # job = qtthread.Job(lambda : collection.load_from_meta(meta_info, info))
            job = qtthread.Job(run)
            jobs_object.add(job)

        return jobs_object
    
    @staticmethod
    def load_from_landmark_meta(collection : DataCollection, meta_info : metadata.LandmarkMeta, lazy_load_flag : bool = True):
        jobs_object = qtthread.Jobs()
        
        def run():
            collection.load_from_meta(meta_info, info)
        for info in meta_info.get_item_iterator():
            # job = qtthread.Job(lambda : collection.load_from_meta(meta_info, info))
            job = qtthread.Job(run)
            jobs_object.add(job)

        return jobs_object



    def save(collection : DataCollection, meta_info : metadata.LandmarkMeta):
        if isinstance(meta_info, metadata.LandmarkMeta):
            return DataIOFactory.save_from_landmark_meta(collection, meta_info)
        elif isinstance(meta_info, metadata.ImageMeta):
            return DataIOFactory.save_from_image_meta(collection, meta_info)


    @staticmethod
    def save_from_landmark_meta(collection : DataCollection, meta_info : metadata.LandmarkMeta, location : str = None ):
        jobs = qtthread.Jobs()
        extension = meta_info.extension
        for info in meta_info : 
            lmk_name =  info.name
            try : 
                uuid = info.unique_id
                f = lambda : save_landmark(collection[uuid].m_landmark.landmark, info.name)
                jobs.add(qtthread.Job(f))
            except:
                pass 
        return jobs
            


    
    def save_from_image_meta(collection : DataCollection, meta_info : metadata.LandmarkMeta, location : str = None):
        jobs = qtthread.Jobs()
        if location is None :
            location = meta_info.file_location
        for info in meta_info : 
            uuid = info.unique_id
            try : 
                f = lambda : save_landmark(collection[uuid].m_landmark.landmark, info.name)
                jobs.add(qtthread.Job(f))
            except : 
                pass 
        return jobs





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





class DataManager:
    def __init__(self, worker : qtthread.Worker):
        self.m_detector = Detector() 
        self.m_worker = worker
        self.reset()


    def get_category_iterator(self):
        return self.m_meta.get_category_iterator()

    def get_item_iterator(self):
        return self.m_meta.get_item_iterator()

    def reset(self):
        self.m_data_collection = DataCollection()
        self.m_current_selected_data = None
        self.m_meta = None
        self.m_data_load_flag = False 
        self.m_detector_load_flag = False 
        
    
    def is_loaded(self):
        return self.m_data_load_flag
    
    def is_detector_loaded(self):
        return self.m_detector.m_is_loaded

    def load_detector(self, pth  =None ):
        self.m_detector.set_path(pth)
        load_job = LoadDetectJob(self.m_detector)
        self.m_worker.reserve_job(load_job)



    def get_data_collection(self):
        return self.m_data_collection
    
    def __len__(self):
        return len(self.m_data_collection)
    
    def __getitem__(self, key_o_idx):
        pass


    def get_selected_data(self):
        return self.m_current_selected_data 
    
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

        if isinstance(self.m_meta, metadata.ImageMeta):
            self.m_meta = self.m_meta.convert_to()
        print(meta.meta_type, " is loaded ...")
        jobs = DataIOFactory.load( self.m_data_collection, self.m_meta, False)
        

        jobs.then(self.__meta_load_finished_callback)
        self.m_worker.reserve_job(jobs, )

    
    def __meta_load_finished_callback(self, x):
        self.m_current_selected_data = self.m_data_collection[0].unique_id

    def detect_all_landmark(self):
        jobs = qtthread.Jobs()

        if self.m_detector.m_is_loaded:
            raise Exception("Detector Not Loaded")

        for data in self.m_data_collection:
            jobs.add(LandmarkDetectJob(data, self.m_detector))
        self.m_worker.reserve_job(jobs)
    def detect_landmark(self, index_o_uuid_key):
        if self.m_detector.m_is_loaded:
            raise Exception("Detector Not Loaded")
        
        job = LandmarkDetectJob(self.m_data_collection[index_o_uuid_key], self.m_detector)
        self.m_worker.reserve_job(job)


    def save_data(self, pth):
        """
            save landmark data
            reserver to worker
        """
        if not osp.exists(pth):
            os.makedirs(pth)
        self.m_meta.write_meta(pth) # save meta

        jobs = DataIOFactory.save(self.m_meta) # save data
        self.m_worker.reserve_job(jobs)
    



if __name__ == "__main__":
    a = qtthread.Worker(None)
    a.start()
    manger = DataManager(a)
    pt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_image")
    manger.load_data_from_meta(pt)
    print(manger)
    while True:
     True