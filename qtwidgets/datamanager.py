

import os 
import os.path as osp

import yaml
import metadata
import image
import qtthread
import image
import flandmark
import uuid 
import numpy as np 
from typing import Union
import signals
import ict_fact_meta
import logger
import math 


data_logger =logger.root_logger.getChild("datamanager")


def save_landmark(lmk, path):
    root_path = osp.dirname(path)
    # print("root", root_path)
    data_logger.debug("save landmark.")
    data_logger.debug("location %s", root_path)
    data_logger.debug("file name %s", osp.basename(path))
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
    
    class DataItemIterator():
        def __init__(self, item_dict:dict):
            self.m_data_collection = list(item_dict.items())
            self.m_cur_idx = 0
            

        def __iter__(self):
            return self
        def __next__(self):
            try : 
                item = self.m_data_collection[self.m_cur_idx][-1]
                self.m_cur_idx += 1
                return item
            except:
                raise StopIteration()


    class Data():
        def __init__(self, uuid):
            self.m_uuid = uuid
            self.m_image = None 
            self.m_lmk =  flandmark.Landmark()

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
        return DataCollection.DataItemIterator(self.m_data_item)

    def __len__(self):
        return len(self.m_data_item.items())
    

    def index_to_key(self, index):
        key = list(self.m_data_item.keys())[index]
        return key

    def key_to_index(self, uuid_name):
        self.m_data_item[uuid_name] # check for raise error.

        for i, key in enumerate(self.m_data_item.keys()):
            if uuid_name == key :
                return i
        
    
    
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
        data.m_lmk = flandmark.Landmark()
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
        def run(info):
            def wrapper():
                collection.load_from_meta(meta_info, info)
        location = meta_info.file_location
        for info in meta_info.get_item_iterator():
            # job = qtthread.Job(lambda : collection.load_from_meta(meta_info, info))
            job = qtthread.Job(run(info))
            jobs_object.add(job)

        return jobs_object
    
    @staticmethod
    def load_from_landmark_meta(collection : DataCollection, meta_info : metadata.LandmarkMeta, lazy_load_flag : bool = True):
        jobs_object = qtthread.Jobs()
        
        def run(info):
            def wrapper():
                collection.load_from_meta(meta_info, info)
            return wrapper
        for info in meta_info.get_item_iterator():
            # job = qtthread.Job(lambda : collection.load_from_meta(meta_info, info))
            job = qtthread.Job(run(info))
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
        path = meta_info.file_location
        if osp.isfile(meta_info.file_location):
            path = osp.dirname(path)
        def save_wrapper(data, landmark_pth):
            def wrapper():
                save_landmark(data, landmark_pth)
            return wrapper
        for info in meta_info.get_item_iterator() : 
            lmk_name =  info.name
            try : 
                uuid = info.unique_id
                data_logger.debug(uuid)
                data_logger.debug(collection[uuid].m_lmk.landmark is None)
                # save_landmark(collection[uuid].m_lmk.landmark, osp.join(path, info.landmark))
                f = save_wrapper(collection[uuid].m_lmk.landmark, osp.join(path, info.landmark))
                jobs.add(qtthread.Job(f))
            except:
                pass 
        data_logger.info("reserver jobs :" )
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
        try : 
            import dlib
            if self.m_path == None :
                self.m_path = "./shape_predictor_68_face_landmarks.dat"
            self.m_detector = dlib.get_frontal_face_detector()
            self.m_predictor = dlib.shape_predictor(self.m_path) 
            self.m_is_loaded = True

        except :
            self.m_is_loaded = False
        

    def detect(self, data_object: DataCollection.Data, landmark_meta_info : ict_fact_meta.BaseFaceMeta):
        
        lmk_size = len(landmark_meta_info)
        h,w = data_object.m_image.shape
        img, ratio = data_object.m_image.resize()
        rects = self.m_detector(img, 1)
        if len(rects) == 0 :
            w,h = data_object.m_image.shape
            v = landmark_meta_info.get_template_mesh()
            mean_v = np.mean(v, axis=0)
            centered_v =  v  - mean_v 
            max_y = np.max(np.abs(centered_v[:, 0]).reshape(-1))
            max_x = np.max(np.abs(centered_v[:, 1].reshape(-1)))
            max_z = np.max(centered_v[:, -1].reshape(-1))

            max_length = max(max_y, max_x)
            max_length += 0.1
            near = 1.0
            far = 1000
            aspect_ratio = w/h
            


            #normal coord porj
            proj = np.array([[near/max_length, 0,  0, 0 ],
                               [0, -near/max_length, 0, 0 ],
                                [0, 0, -(far+near)/(far-near), -2*(far*near)/(far-near) ],
                                [0,0,-1,0]])            #image coord proj
            
            aspect_applied_w = w
            img_proj = np.array([[aspect_applied_w, 0,  aspect_applied_w/2 ],
                                [0, h, h/2 ],
                                [0, 0, 1]])
            cam_trans = max_z
            centered_v[:, -1] -= (cam_trans + 10.0)
            centered_v_h = np.concatenate([centered_v, np.ones((len(centered_v), 1), dtype=np.float32) ], axis=-1)
            proj_2d = (proj @centered_v_h.T).T
            proj_2d[:, :-1] /= proj_2d[:, -1].reshape(-1,1)
            proj_2d = proj_2d[:, :-1]
            proj_2d[:, -1] = 1.0
            proj_img_h = (img_proj@proj_2d.T).T
            proj_img = proj_img_h[:, :-1]

            data_object.m_lmk.landmark = proj_img

            return 


        for i, rect in enumerate(rects):
            l = rect.left()
            t = rect.top()
            b = rect.bottom()
            r = rect.right()
            shape = self.m_predictor(img, rect)
            data_object.m_lmk = flandmark.Landmark()
            data_object.m_lmk.landmark = np.empty((lmk_size, 2),np.float32)
            for j in range(lmk_size):
                x, y = shape.part(j).x*ratio, shape.part(j).y*ratio
                data_object.m_lmk.landmark[j, :] = (x, y)
    

class LandmarkDetectJob(qtthread.Runnable):
    def __init__(self, data_object : DataCollection.Data, detector : Detector, landmark_meta_info : ict_fact_meta.BaseFaceMeta):
        super().__init__()
        self.data_object = data_object
        self.detector = detector
        
        self.landmark_meta_info = landmark_meta_info

    def run(self):
        self.detector.detect(self.data_object, self.landmark_meta_info)


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


    def get_meta(self):
        return self.m_meta

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
        return self.m_data_collection[key_o_idx]

    def set_selected_data_uuid(self, uuid):
        self.m_current_selected_data = uuid
        return self.m_current_selected_data
    def set_selected_data_from_index(self, index):
        key = self.m_data_collection.index_to_key(index)
        self.m_current_selected_data = key
        
        return self.m_current_selected_data


    def get_selected_data_uuid(self):
        return self.m_current_selected_data 
    
    def get_selected_data_index(self):
        return self.m_data_collection.key_to_index(self.m_current_selected_data)



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
        self.m_worker.reserve_job(jobs, signals.Event(signals.EventType.DATA_LOADED_FROM_META))

    
    def __meta_load_finished_callback(self, x):
        self.m_current_selected_data = self.m_data_collection[0].unique_id

    def detect_all_landmark(self):
        jobs = qtthread.Jobs()

        if not self.m_detector.m_is_loaded:
            raise Exception("Detector Not Loaded")

        for data in self.m_data_collection:
            # LandmarkDetectJob(data, self.m_detector, self.get_landmark_structure_meta())._run()
            jobs.add(LandmarkDetectJob(data, self.m_detector,self.get_landmark_structure_meta()))
        self.m_worker.reserve_job(jobs, job_finish_event_type=signals.Event(signals.EventType.ALL_LANDMARK_DETECTED))

    def detect_landmark(self, index_o_uuid_key):
        if not self.m_detector.m_is_loaded:
            raise Exception("Detector Not Loaded")
        
        job = LandmarkDetectJob(self.m_data_collection[index_o_uuid_key], self.m_detector)
        self.m_worker.reserve_job(job, job_finish_event_type=signals.Event(signals.EventType.ALL_LANDMARK_DETECTED))


    def save_data(self, pth):
        """
            save landmark data
            reserver to worker
        """
        if not osp.exists(pth):
            os.makedirs(pth)
        self.m_meta.write_meta(pth) # save meta

        jobs = DataIOFactory.save(self.m_data_collection, self.m_meta) # save data
        self.m_worker.reserve_job(jobs)
        data_logger.debug("reserve save_data on thread.")
    

    def load_landmark_meta(self, pth):
        self.m_landmark_structure_meta = ict_fact_meta.IctFaceMeta()
        self.m_landmark_structure_meta.load_from_file(pth)
        data_logger.debug("load landmark structure meta %s", pth)


    def get_landmark_structure_meta(self):
        if hasattr(self, "m_landmark_structure_meta"):
            return self.m_landmark_structure_meta
        raise Exception("File Not Loaded Error.")




if __name__ == "__main__":
    a = qtthread.Worker(None)
    a.start()
    manger = DataManager(a)
    pt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_image")
    manger.load_data_from_meta(pt)
    print(manger)
    while True:
     True