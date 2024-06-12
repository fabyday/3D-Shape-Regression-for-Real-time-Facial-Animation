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


class Worker(QThread):
    def __init__(self):
        self.jobs = []
        self.mutex = QMutex()

    def run(self, runnable):
        
        pass







if __name__ == "__main__":


    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    form = ConfigurationEditor()
    form.show()
    exit(app.exec_())