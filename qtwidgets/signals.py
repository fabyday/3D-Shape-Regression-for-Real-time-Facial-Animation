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


class SignalCollection(QObject):
    InspectorIndexSignal = pyqtSignal(int)
    InspectorLmkDetectSignal = pyqtSignal(int)
    InspectorSaveSignal = pyqtSignal(int)
    InspectorLoadMetaData = pyqtSignal()




if __name__ == "__main__":


    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    form = ConfigurationEditor()
    form.show()
    exit(app.exec_())