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
import time
import yaml
import queue
class Runnable():
    def __init__(self):
        pass 
    def run(self):
        return NotImplemented

class Worker(QThread):
    def __init__(self, parent):
        """
            parent : UI Object
        """
        self.jobs_queue = queue.Queue()
        self.m_parent = parent
        self.m_mutex = QMutex()
    def reserve_job(self, job_o_joblist : Runnable | list ):

        self.m_mutex.lock()
        if isinstance(job_o_joblist, Runnable):
            self.jobs_queue.put(job_o_joblist)
        self.m_mutex.unlock()

    def run(self):
        while True : 
            time.sleep(0.0001)
            self.m_mutex.lock()
            data = self.jobs_queue.get()
            self.m_mutex.unlock()
            data.run()

class DummyJob(Runnable):
    def __init__(self):
        pass

    def run(self):
        print('test')
class LandmarkDetectJob(Runnable):
    def __init__(self):
        pass 
    

    def run(self):
        pass 


class ImageLoadJob(Runnable):
    def __init__(self):
        pass
    def run(self):
        pass








if __name__ == "__main__":
    a = Worker()
    a.start()

    # from PyQt5.QtWidgets import QApplication

    # app = QApplication(sys.argv)
    # form = ConfigurationEditor()
    # form.show()
    # exit(app.exec_())