

import typing
from PyQt5.QtWidgets import *
import time
from PyQt5 import QtGui, QtCore 
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt

from PyQt5.QtGui import QImage 
from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal, QThread, QMutex
import sys
import ctypes
import os.path as osp
import os 
import cv2
import yaml
import time 
import copy

mediapipe_based_metadata_preset = {}

metadata_default_filename = "meta.yaml"


#define fowrad
class EditorMeta:
    pass

class RecentEditorMeta:
    default_max_size = 10
    class MetaKey:
        type = "type"
        recent_editor_meta = "recent_editor_meta"
        visited_time = "visited_time"

    class EditorMetaItem:
        def __init__(self, name, location, visited_time):
            self.name = name 
            self.location = location
            self.visited_time = visited_time

    def __init__(self):
        self.type = "recent_meta"
        self.meta_list = []
        self.max_size = RecentEditorMeta.default_max_size
        

    def set_max_size(self, size : int):
        if size <= 0 : 
            size = RecentEditorMeta.default_max_size
        self.max_size = size 

    def push_editor_meta(self, meta_o : EditorMeta):
        meta_name = meta_o.get_metadata_name()
        meta_loc = meta_o.get_metadata_location()


        if meta_loc:
            mm = RecentEditorMeta.EditorMetaItem(name=meta_name,location=meta_loc, visited_time=time.time())


    
    def get_recent_editor_meta(self):
        return self.meta_list

    def load(self):
        pass

    def save(self):
        data = dict()
        data[RecentEditorMeta.MetaKey.type] = self.type
        data[RecentEditorMeta.MetaKey.recent_editor_meta] = []



#  __import__



class MetaNotFoundException(Exception):
    def __init__(self, *args):
        super(Exception, self).__init__(*args)
        # self.meta = 


class EditorMeta:
    class MetaKey:
        type = "type"
        editor_meta_location = "editor_meta_location"
        
        width = "width"
        height = "height"
        default_meta_location = "root_meta_loc"

        dectector_meta_location = "dectector_meta_location"
        image_meta_location = "image_meta_location"
    
    class MetaReduplicationException(Exception):
        def __init__(self, *args):
            super(Exception, self).__init__(*args)

    default_editor_meta_path = "./editor"

        
    def __init__(self):
        self.type = "Editor Meta"
        self.metadata_name = None
        self.metadata_location = None 
        
        self.image_metadata = None
        self.detector_meta = None 
        self.recent_meta = None

    def clone(self, desired_name):
        reval = copy.deepcopy(self)
        reval.metadata_name[EditorMeta.MetaKey.editor_meta_location] = desired_name
    
    def get_type(self):
        return self.type
    
    def get_metadata_name(self):
        return self.metadata_name
    
    def get_metadata_location(self):
        return self.metadata_location

    def get_image_metadata(self):
        return self.image_metadata
    
    def get_detector_meta(self):
        return self.detector_meta
    
    def load(self, o):
        if isinstance(o, str):
            with open(o, "r") as fp:
                meta_infos = yaml.load(o, yaml.FullLoader)
            meta_type = meta_infos[EditorMeta.MetaKey.type]
            if meta_type == self.type:
                self.metadata_location = meta_infos[EditorMeta.MetaKey.editor_meta_location]
                img_meta_location = meta_infos[EditorMeta.MetaKey.image_meta_location]
                detector_meta_location = meta_infos[EditorMeta.MetaKey.dectector_meta_location]

                imeta = ImageMeta()
                imeta.load(img_meta_location)
                imeta.set_editor_meta_object(self)

                dmeta = DetectorMeta()
                dmeta.load(detector_meta_location)
                dmeta.set_editor_meta_object(self)
                self.image_metadata = imeta 
                self.detector_meta = dmeta
                
                
            else : # this creat empty meta data 
                ii = 0
                self.metadata_name = "new meta"
                while True:
                    filename, ext = osp.splitext(metadata_default_filename)
                    if ii == 0 :
                        addi = ""
                        ii += 1
                    else:
                        addi = str(ii)
                        ii += 1
                    candidate_location = osp.join(EditorMeta.default_editor_meta_path, filename+addi + ext)
                    if not osp.exists(candidate_location):
                        self.metadata_location = candidate_location
                        break
                
                imeta = ImageMeta()
                imeta.load(o)
                imeta.set_editor_meta_object(self)

                dmeta = DetectorMeta()
                dmeta.load(o)
                dmeta.set_editor_meta_object(self)
                self.image_metadata = imeta 
                self.detector_meta = dmeta


class ImageMeta:
    class MetaKey:
        type = "type"
        image_root_location = "image_root_location"
        image_metadata_location = "image_metadata_location"
        category_list = "category_list"
        

    def __init__(self) -> None:
        self.type = "Image Meta"
        self.image_root_location = None 
        self.image_metadata_location = None 
        self.category_list = []

        self.editor_metadata = None 

    def set_editor_meta_object(self, o :EditorMeta):
        self.editor_metadata = o 
    def load(self, o):
        if isinstance(o, str):
            with open(o, "r") as fp:
                meta_infos = yaml.load(o, yaml.FullLoader)
            meta_type = meta_infos[EditorMeta.MetaKey.type]
        elif isinstance(o, dict):
            o[ImageMeta.MetaKey.category_list]
        else:
            raise MetaNotFoundException("Image Meta initialization was not succeed.")

class DetectorMeta:
    class MetaKey:
        type = "type"
        detector_location = "detector_location"
        detector_metadata_location = "detector_metadata_location"
        detector_name = "detector_name"
    def __init__(self):
        self.m_type = "Detector Meta"
        self.m_detector_name = None 
        self.m_detector_metadata_location = None
        self.m_detector_location = None 
        
        self.m_editor_metadata_object = None 
    
    @property
    def dectector_name(self):
        return self.m_detector_name
    
    @dectector_name.setter
    def detector_name(self, name : str):
        self.m_detector_name = name

    @property
    def detector_location(self):
        return self.m_detector_location
    
    @detector_location.setter
    def detector_location(self, path):
        self.m_detector_location = path
    @property
    def detector_metadata_location(self):
        return self.detector_metadata_location
    @detector_metadata_location.setter
    def detector_metadata_location(self, arg:str):
        self.m_detector_metadata_location = arg 

    @property
    def editor_metadata_object(self):
        return self.m_editor_metadata_object

    @editor_metadata_object.setter
    def editor_metadata_object(self, o : EditorMeta):
        self.m_editor_metadata = o
    
    @property
    def type(self):
        return self.m_type

    def load(self, o, editor_meta = None):
        if isinstance(o, str):
            with open(o, "r") as fp:
                o = yaml.load(o, yaml.FullLoader)
            meta_type = o.get(EditorMeta.MetaKey.type, None)
            
        elif isinstance(o, dict):
            meta_type = o.get(EditorMeta.MetaKey.type, None)
        else:
            raise MetaNotFoundException("DetectorMeta initialization was not succeed.")

        if meta_type == self.m_type:
            self.editor_metadata_object = editor_meta
            self.detector_name = o.get(DetectorMeta.MetaKey.detector_name)
            self.detector_location = o.get(DetectorMeta.MetaKey.detector_location)
            self.detector_metadata_location =     o.get(DetectorMeta.MetaKey.detector_metadata_location)
            self.succ_load = True

    def __bool__(self):
        if hasattr(self, "succ_load"):
            return True
        return False

    def __nonzero__(self):
        return self.__bool__()

class MetaDataManager:
    def __init__(self):
        self.editor_metadatas = []

    def load_metadata(self, o : str or dict):
        editor_meta = EditorMeta()
        try :
            editor_meta.load(o)
        except MetaNotFoundException as e :
            print(e.with_traceback()) 
            editor_meta = EditorMeta()
        finally:
            self.editor_metadatas.append(editor_meta)
        
            
class GlobalData:
    def __init__(self, metas, img, detector):
        pass

class ConfigurationEditor(QWidget):
    def __init__(self, global_infos=None, worker_obj = None):
        QWidget.__init__(self, flags=Qt.Widget)

        self.worker_queue = worker_obj
        self.setWindowTitle("Configuration Editor")
        

    def init_ui(self):
        pass

    
        


if __name__ == "__main__":


    test_detector_data = {"type" : "Detector Meta",
        "detector_location" :"test_loc/test.txt",
        "detector_metadata_location" : "test_meta/meta.txt",
        "detector_name" :"new_meta" }
    
    
    class MetaReduplicationException(Exception):
        def __init__(self, *args):
            super(Exception, self).__init__(*args)

    default_editor_meta_path = "./editor"
    test_editor_data =  {"type" : "Editor Meta", 
                        "editor_meta_location" : "editor/meta.txt",
                        "width" : 1000,
                        "height" : 2000,
                        "root_meta_loc" : "root/meta.txt",
                        "dectector_meta_location" : "meta_loca",
                        "image_meta_location" : "image/meta.txt"                     
                      }
    test_image_meta = {
        "type" : "Image Meta"
        "image_root_location" : "image/root"
         "image_metadata_location" : "image/root/location"
        "category_list" : ["test1",]
        

    def __init__(self) -> None:
        self.image_root_location = None 
        self.image_metadata_location = None 
        self.category_list = []

        self.editor_metadata = None 
    }


    from PyQt5.QtWidgets import QApplication
    meta = MetaDataManager()
    gdata =GlobalData()
    app = QApplication(sys.argv)
    form = ConfigurationEditor(global_infos=gdata, worker = None)
    form.show()
    exit(app.exec_())