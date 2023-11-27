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
from . import qtthread
from qtwidgets import metadata

class Detector():


    def __init__(self):
        self.detector_f = None 
        self.detector_load_f = lambda x : (False, "Not Impl")


    # do not use it directly.
    def load_detector(self):
        """
        return init status, and description
        """
        return self.detector_load_f()


    def detect_lmk(self, img):
        return self.detector_f(img)


    def detector_setup(self):
        pass
