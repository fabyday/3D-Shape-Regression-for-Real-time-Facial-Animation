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