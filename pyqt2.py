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
class ImageMeta:
    pass 

class ImgFileExt(enum.Enum):
    JPG = "JPG"
    PNG = "PNG"

class Image:
    def __init__(self):
        pass 

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
    pass 


class ImageCollection():
    def __init__(self):
        pass






    def save_data_by_name():
        pass 


    def save_data_by_index():
        pass 


    def save_data_by_group():
        pass

class InspectorWidget(QWidget):
    def __init__(self, ctx):
        super().__init__()

        self.m_ctx = ctx
        self.m_root_dir = [QLineEdit(""), QPushButton("...")]
        self.m_save_root_loc = [QLineEdit(""), QPushButton("...")]
        
        self.root_dir_widget.setReadOnly(True)
        self.save_root_loc.setReadOnly(True)


        self.m_meta_file = QLineEdit("Not Found.")
        self.m_meta_file.setReadOnly(True)
        
        self.m_image_name = QLineEdit("Empty")
        self.m_image_name.setReadOnly(True)
        
        self.m_category = QLineEdit("Empty")
        self.m_category.setReadOnly(True)

        self.m_prev_btn = QPushButton("Prev")
        self.m_next_btn = QPushButton("Next")



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

        layout_info.addRow('root_directory', layout1)
        layout_info.addRow('save root location', layout2)

        for i, (key, item) in enumerate(self.data.items()):
            layout_info.addRow(key, item)



        image_control_layout = QVBoxLayout()
        image_number_info_layout = QHBoxLayout()

        image_control_layout.setContentsMargins(-1, 0, -1, -1)
        image_number_info_layout.setContentsMargins(-1, 0, -1, -1)


        image_number_info_layout.addWidget(self.image_info[0])
        image_number_info_layout.addWidget(self.image_info[1])
        image_number_info_layout.addWidget(self.image_info[2])
        image_number_info_layout.addWidget(self.image_info[3])

        image_move_layout = QHBoxLayout()
        image_move_layout.addWidget(self.prev)
        image_move_layout.addWidget(self.next)
        image_control_layout.addLayout(image_number_info_layout)
        image_control_layout.addLayout(image_move_layout)

        button_layout.setContentsMargins(-1,0,-1,-1)
        button_layout.addWidget(self.detect_button,0, 0)
        button_layout.addWidget(self.save_individual_button,0, 1)
        button_layout.addWidget(self.save_all_button,1, 0, 1, 2)
        button_layout.addWidget(self.detect_all_button,2,0,1,2)
        
        
        main_layout.addLayout(layout_info)
        main_layout.addLayout(image_control_layout)
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        
        self.show() 

    @property
    def root_dir_widget(self):
        return self.m_root_dir[0]
    
    @property
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