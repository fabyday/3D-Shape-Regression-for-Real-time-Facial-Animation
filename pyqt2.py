import typing
from PyQt5.QtWidgets import *
import time
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QImage
from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal, QThread, QMutex
import sys
import ctypes
import os.path as osp
import os 
import cv2
import yaml
import face_main_test as ff
import dlib 
import enum
DEFAULT_PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"


async def async_load_image(path):
    return cv2.imread(path)
def load_image(path):
    return cv2.imread(path)


def check_and_load_meta_file(root_pth_name):
    if osp.exists(osp.join(root_pth_name, "meta.yaml")):
        self.data['meta file'].setText("meta.yaml")
        j_id = int(time.time())
        self.jobs[j_id] = False
        cancel_f = self.worker.do_load_image(j_id, [self.program_data.get_cur_index()])
        self.current_job_item = (j_id , cancel_f)
        self.upate_ui()
        self.signal.InspectorLoadMetaData.emit()
        self.worker.load_detector()
    else:
        self.data["meta file"].setText("Not Found.")

def open_and_find_directory(Qt_object, callback):
    # callback input is file name
    #https://stackoverflow.com/questions/64530452/is-there-a-way-to-select-a-directory-in-pyqt-and-show-the-files-within
    def wrapper():
        dialog = QFileDialog(Qt_object, windowTitle='Select Root Directory')
        dialog.setDirectory(osp.abspath(os.curdir))
        dialog.setFileMode(dialog.Directory)
        dialog.setOptions(dialog.DontUseNativeDialog)        
        fileTypeCombo  = dialog.findChild(QComboBox, 'fileTypeCombo')

        if fileTypeCombo:
            fileTypeCombo.setVisible(False)
            dialog.setLabelText(dialog.FileType, '')
    
        if dialog.exec_():
            callback(dialog.selectedFiles()[0])
        else :
            callback(None)
    return wrapper

def open_and_find_meta(Qt_object, callback):
    # callback input is file name
    #https://stackoverflow.com/questions/64530452/is-there-a-way-to-select-a-directory-in-pyqt-and-show-the-files-within
    def wrapper():
        dialog = QFileDialog(Qt_object, windowTitle='Select meta file')
        dialog.setDirectory(osp.abspath(os.curdir))
        dialog.setFileMode(dialog.ExistingFile)
        dialog.setNameFilter("*.yaml")
        dialog.setOptions(dialog.DontUseNativeDialog)        
        fileTypeCombo  = dialog.findChild(QComboBox, 'fileTypeCombo')
        
        if fileTypeCombo:
            fileTypeCombo.setVisible(False)
            dialog.setLabelText(dialog.FileType, '')
    
        if dialog.exec_():
            callback(dialog.selectedFiles()[0])
        else :
            callback(None)
    return wrapper



class DefaultDetector():
    def __init__(self):
        self.img_size = (None, None)


    @property
    def detect_img_size(self):
        return self.img_size

    @detect_img_size.setter
    def detect_img_size(self, width_height_tuple):
        self.img_size = width_height_tuple

    
    def load(self, path=None):
        if path == None :
            path = DEFAULT_PREDICTOR_PATH
        self.detector = dlib.get_frontal_face_dtector()
        self.predictor = dlib.shape_predictor(path)


    def predict(self, img):
        if not (hasattr(self, "predictor") and hasattr(self, "detector")):
            raise Exception("detector and predictor was not loaded.")

        if len(img.shape) == 2:
            h,w  = img.shape
        else : 
            h,w, _ = img.shape
        

        rects = self.detector(img, 1)
        for i, rect in enumerate(rects):
            shape = self.predictor(img, rect)
            for j in range(68):
                    x, y = shape.part(j).x, shape.part(j).y
                    self.lmk[j][0] = x
                    self.lmk[j][1] = y

class UncompatibleMetaData(Exception):
    pass 

class AbstractMeta:
    def __init__(self, type_name : str):
        self.type_name = type_name
        self.loaded = False
        pass 
    def save(self, path):
        raise NotImplemented
    
    def load(self, path):
        raise NotImplemented 
    


class IctLandmarkMeta(AbstractMeta):
    META_NAME = "ict_landmark_meta"
    def __init__(self):
        super().__init__(self, IctLandmarkMeta.META_NAME)


    def save(self, path):
        raise NotImplemented

    def load(self, path):
        if not path == None:
            pass

def meta_load_helper(self, path):
    meta = None 
    for meta_class in [InputMeta, IctLandmarkMeta, OutputMeta]:
        meta = InputMeta()
        try : 
            meta.load(path)
        except UncompatibleMetaData :
            continue
    
    if not meta : # if meta is None 
        raise UncompatibleMetaData
    return meta


class InputMeta(AbstractMeta):
    META_NAME = "input_meta"
    def __init__(self):
        super().__init__(self, InputMeta.META_NAME)
        self.m_ext = ""
        self.m_meta = ""
        self.m_ext = ""
        self.m_src_path = ""
        self.m_dest_path = ""
    
    def save(self, path):
        root_dir = os.path.dirname(path)
        self.m_dest_path = path
        if not os.path.exists(root_dir):
            os.makedirs(path)
        
        res = {"meta" : dict()}

        meta = res["meta"]
        meta['file_ext'] = self.m_ext
        with open(path, 'w') as fp:
            self.m_keys()
            self.m_meta
            meta["images_name"] = 
            self.m_keys = raw_meta["meta"]["images_name"].keys()
            self.m_meta = raw_meta["meta"]["images_name"]
            self.m_ext =  raw_meta["meta"]["file_ext"]
            self.loaded = True

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError
        
        self.m_src_path = path
        with open(path, 'r') as fp:
            raw_meta = yaml.load(fp, yaml.FullLoader)
        try :
            self.m_keys = raw_meta["meta"]["images_name"].keys()
            self.m_meta = raw_meta["meta"]["images_name"]
            self.m_ext =  raw_meta["meta"]["file_ext"]
            self.loaded = True
        except : 
            raise UncompatibleMetaData

class OutputMeta(AbstractMeta):
    META_NAME = "output_meta"
    class landark_image_tuple:
        def __init__(self, lmk_name, img_name) -> None:
            self.m_lmk_name = lmk_name 
            self.m_img_name = img_name

        @property
        def lmk_name(self):
            return self.m_lmk_name 
        
        @property
        def image_name(self):
            return self.m_img_name 
        


    def __init__(self, file_ext="JPG"):
        super().__init__(self, OutputMeta)
        self.m_file_ext = file_ext
        self.m_image_root_directory = ""
        self.m_landmark_root_directory = ""
        self.m_category = dict()

        self.m_landmark_meta_file_name = ""

    def load(self, path):
        with open(path) as fp :
            meta_stream = yaml.load(fp) 
        meta_stream['meta']

    def save(self, path):
        pass

    def add(self, category, lmk_name, img_name):
        list = self.m_category.get(category, [])
        list.append(OutputMeta.landark_image_tuple(lmk_name, img_name))

    def remove(self):
        pass

    @property
    def landmark_meta_file(self):
        return self.m_landmark_meta_file_name

    @landmark_meta_file.setter
    def landmark_meta_file(self, meta_pth):
        if os.path.exists(meta_pth):
            self.m_landmark_meta_file_name = meta_pth
        else :
            raise FileNotFoundError
        


class ImgFileExt(enum.Enum):
    JPG = "JPG"
    PNG = "PNG"

class Image:
    def __init__(self):
        pass 

    async def load(self, pth):
        self.m_image = cv2.imread(pth)
        self.m_name, self.m_ext = osp.splitext(osp.basename(pth))
    
    async def save(self, pth):
        cv2.imwrite(self.m_iamge, pth)

    @property 
    def image(self):
        return self.m_image 
    
    @image.setter
    def image(self, image):
        self.m_image = image
    

    @property
    def extension(self):
        return self.extension
    
    @extension.setter
    def extension(self, ext : ImgFileExt):
        self.ext = ext


    @property
    def name(self):
        if hasattr(self, "m_name"):
            return self.m_name
        return ""
    @name.setter
    def name(self, name):
        self.m_name = name 
    @property
    def landmark(self):
        if hasattr(self, "m_landmark"):
            return self.m_landmark
        return None
    
    
    @landmark.setter 
    def landmark(self, lmk):
        self.m_landmark = lmk
    

    

class ProgramContext:

    def __init__(self) :
        self.m_input_meta_object = None 
        self.m_output_meta_object = None 
        self.m_landmark_meta_object = None 
        self.m_image_collection_object = None 



class Worker(QThread):
    
    def __init__(self, parent):
        pass

    def run():
        pass
    
class InspectorWidget(QWidget):

    detector_loaded = pyqtSignal(bool)

    def __init__(self, ctx: ProgramContext):
        super().__init__()

        self.m_ctx = ctx
        self.m_root_dir = [QLineEdit(""), QPushButton("...")]
        self.m_save_root_loc = [QLineEdit(""), QPushButton("...")]


        self.m_landmark_meta = [QLineEdit("Not Found"), QPushButton("...")]
        self.m_landmark_meta[0].setReadOnly(True)
        
        self.m_root_dir[0].setReadOnly(True)
        self.m_save_root_loc[0].setReadOnly(True)

        # filename form
        self.m_meta_file = QLineEdit("Not Found.")
        self.m_meta_file.setReadOnly(True)
        # image name form
        self.m_image_name = QLineEdit("Empty")
        self.m_image_name.setReadOnly(True)
        # category form
        self.m_category = QLineEdit("Empty")
        self.m_category.setReadOnly(True)
        #prev next button form
        self.m_prev_btn = QPushButton("Prev")
        self.m_next_btn = QPushButton("Next")

        self.m_image_info  = [QLabel("image number"), QLineEdit("?"), QLabel("/"), QLabel("?")]

        
        self.m_detect_button = QPushButton("detect")
        self.m_save_individual_button  = QPushButton("save_data")
        self.m_save_all_button = QPushButton("save_all")
        self.m_detect_all_button = QPushButton("detect all lmk from entire images")


    def setup_save_loc(self):
        # self.open_and_find_directory()
        pass

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(-1,0,-1,-1)
        main_layout.setAlignment(QtCore.Qt.AlignTop)
        layout_info = QFormLayout()
        button_layout = QGridLayout()

        layout1 = QHBoxLayout()
        layout2 = QHBoxLayout()
        layout1.addWidget(self.m_root_dir[0])
        layout1.addWidget(self.m_root_dir[1])

        layout2.addWidget(self.m_save_root_loc[0])
        layout2.addWidget(self.m_save_root_loc[1])
        
        layout1.setContentsMargins(-1, 0, -1, -1)
        layout2.setContentsMargins(-1, 0, -1, -1)

        layout_info.addRow('Root directory', layout1)
        layout_info.addRow('Save root location', layout2)

        layout3 = QHBoxLayout()
        layout3.addWidget(self.m_landmark_meta[0])
        layout3.addWidget(self.m_landmark_meta[1])
        layout_info.addRow("Meta file", self.m_meta_file)
        layout_info.addRow("Landmark Meta file", layout3)
        layout_info.addRow("Image name", self.m_image_name)
        layout_info.addRow("Category", self.m_category)



        image_control_layout = QVBoxLayout()
        image_number_info_layout = QHBoxLayout()

        image_control_layout.setContentsMargins(-1, 0, -1, -1)
        image_number_info_layout.setContentsMargins(-1, 0, -1, -1)


        image_number_info_layout.addWidget(self.m_image_info[0])
        image_number_info_layout.addWidget(self.m_image_info[1])
        image_number_info_layout.addWidget(self.m_image_info[2])
        image_number_info_layout.addWidget(self.m_image_info[3])

        image_move_layout = QHBoxLayout()
        image_move_layout.addWidget(self.m_prev_btn)
        image_move_layout.addWidget(self.m_next_btn)
        image_control_layout.addLayout(image_number_info_layout)
        image_control_layout.addLayout(image_move_layout)

        button_layout.setContentsMargins(-1,0,-1,-1)
        button_layout.addWidget(self.m_detect_button,0, 0)
        button_layout.addWidget(self.m_save_individual_button,0, 1)
        button_layout.addWidget(self.m_save_all_button,1, 0, 1, 2)
        button_layout.addWidget(self.m_detect_all_button,2,0,1,2)
        
        
        main_layout.addLayout(layout_info)
        main_layout.addLayout(image_control_layout)
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        self.init_connection()
        self.show() 


    # event emitter
    def next(self):
        pass 

    def prev(self):
        pass

    def detect(self):
        pass

    def save_data(self):
        pass

    def save_data(self):
        pass 

    def save_data_all(self):
        pass 

    def detect_all_lmk_from_entire_images(self):
        pass 
    
    def open_root_dir_and_load(self, pth_name):
        try : 
            self.root_dir_widget = pth_name
        except  FileNotFoundError:
            return # if path was not founded. pass the loading meta.
        

    def open_save_loc(self, pth_name):
        try : 
            self.save_root_loc = pth_name
        except FileNotFoundError :
             self.save_root_loc = ""
    

    def open_meta_file(self, pth):
        available_meta_file_types = [InputMeta, OutputMeta]
        for cls in available_meta_file_types:
            meta_cls = cls()
            try :
                meta_object = meta_cls.load(pth)

            except UncompatibleMetaData:
                meta_object = None 
                continue  

        if  meta_object.type_name == InputMeta.META_NAME: 
            self.m_ctx.m_input_meta_object = meta_object
            return meta_object
        elif  meta_object.type_name == OutputMeta.META_NAME: 
            self.m_ctx.m_input_meta_object = meta_object
            return meta_object
        else:
            raise UncompatibleMetaData
        


            


    def open_lmk_meta_file(self, pth_name):
        try : 
            self.m_landmark_meta[0].setText(pth_name)
            meta = IctLandmarkMeta()
            meta.load(pth_name)
            self.m_ctx.m_landmark_meta_object = meta
        except UncompatibleMetaData:
            self.m_landmark_meta[0].setText("")
            pass 

    #################################################################################################

    def init_connection(self):
        # self.program_data.save_location = self.input_data['save root location'][0].text()
        # self.root_dir_widget.clicked.connect(open_and_find_directory(self, lambda name : self.m_root_dir[0].setText(name)))
        self.root_dir_widget.clicked.connect(open_and_find_directory(self, self.open_root_dir_and_load))
        self.save_root_loc.clicked.connect(open_and_find_directory(self, self.open_save_loc))
        self.m_meta_file
        self.m_landmark_meta[1].clicked.connect(open_and_find_meta(self, self.open_lmk_meta_file))
        
        self.m_prev_btn.clicked.connect(self.prev)
        self.m_next_btn.clicked.connect(self.next)

        # self.m_detect_button.clicked.connect()
        # self.m_detect_all_button.clicked.connect()

        # self.m_save_individual_button.clicked.connect()
        # self.m_save_all_button.clicked.connect()

    @property
    def root_dir_widget(self):
        return self.m_root_dir[-1]
    
    @root_dir_widget.setter
    def root_dir_widget(self, path_name):
        if os.path.exists(path_name):
            self.m_root_dir[0].setText(path_name)
        else:
            raise FileNotFoundError
    @property
    def save_root_loc(self):
        return self.m_save_root_loc[-1]
    
    @save_root_loc.setter
    def save_root_loc(self):
        return self.m_save_root_loc[0]
    
    

class ImageWidget(QGraphicsView):
    def __init__(self, ctx):
        self.scene = QGraphicsScene()
        self.m_ctx = ctx
        super(ImageWidget, self).__init__(self.scene)

    def init_ui(self):
        pass 
class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.m_ctx = ProgramContext()
        self.init_ui()
        

    def init_ui(self):
        width = ctypes.windll.user32.GetSystemMetrics(0)
        height = ctypes.windll.user32.GetSystemMetrics(1)
        self.setWindowTitle('landmark editor')
        self.resize(width*0.8, height*0.8)
        
        imagewidget = ImageWidget(self.m_ctx)

        inspectorwidget = InspectorWidget(self.m_ctx)

        widget = QWidget()

        layout = QHBoxLayout(widget)
        layout.addWidget(imagewidget, 7)
        layout.addWidget(inspectorwidget, 3)
        self.setCentralWidget(widget)
        inspectorwidget.init_ui()

        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_label = QLabel()
        self.statusbar.addWidget(self.status_label)
        self.statusbar.addPermanentWidget(self.progress_bar)
        self.imagewidget = imagewidget
        self.inspectorwidget = inspectorwidget




if __name__ == "__main__":
   app = QApplication(sys.argv)
   ex = MyApp()
   ex.show()
   sys.exit(app.exec_())