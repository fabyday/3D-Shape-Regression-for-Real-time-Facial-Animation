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

class ImageClassWidget():
    pass


class ImageViewWidget(QGraphicsView):
    def __init__(self, program_data: metadata.DataManager, worker : qtthread.Worker):

        self._scene = QGraphicsScene()
        super(ImageWidget, self).__init__(self._scene)
        self.worker = worker

        self.worker.lmk_load_finished.connect(self.reload_image)
        self.worker.lmk_load_finished.connect(self.reload_lmk_to_view)

        self.program_data = program_data

        self.test = QtGui.QPixmap()
        # self.test.load("./test.JPG")
        # self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        # self._scene.addPixmap()
        self.img_overlay = self._scene.addPixmap(self.test)




if __name__ == "__main__":


    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    form = ConfigurationEditor()
    form.show()
    exit(app.exec_())