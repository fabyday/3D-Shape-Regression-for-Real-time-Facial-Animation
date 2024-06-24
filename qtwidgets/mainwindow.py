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
import qtthread
import metadata 
import datamanager
import qtthread
import uuid
import imageviewwidget
import inspectwidget
import signals
class MyApp(QMainWindow):
    
    selected_data_changed_signal = pyqtSignal(uuid.UUID)
    
    data_loaded_signal = pyqtSignal(uuid.UUID)
    
    data_selected_signal = pyqtSignal(uuid.UUID)


    def __init__(self):
        super().__init__()
        self.worker_thread = qtthread.Worker(self)
        self.worker_thread.start()
        self.m_data_manager = datamanager.DataManager(self.worker_thread)

        self.initUI()
        # self.connect_widgets_functionality()
    def initUI(self):
        width = ctypes.windll.user32.GetSystemMetrics(0)
        height = ctypes.windll.user32.GetSystemMetrics(1)
        self.setWindowTitle('landmark editor')
        self.resize(int(width*0.8), int(height*0.8))
        self.move(self.screen().geometry().center() - self.frameGeometry().center())


        widget = QWidget()

        layout = QHBoxLayout(widget)
    #     imagewidget.initUI()
    #     inspectorwidget.initUI()
        self.setCentralWidget(widget)
        self._init_image_view_ui(layout)
        self._init_inspect_ui(layout)
        self._init_status_ui()
        self.m_data_manager.load_detector()

    def _init_inspect_ui(self, layout):
        self.inspectorwidget = inspectwidget.InspectorWidget(self.m_data_manager)
        self.inspectorwidget.init_ui()
        layout.addWidget(self.inspectorwidget, 3)


    def _init_image_view_ui(self, layout):
        self.imagewidget = imageviewwidget.ImageViewWidget(self, self.m_data_manager)
        # imagewidget
        layout.addWidget(self.imagewidget, 7)
    

    def _init_status_ui(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_label = QLabel()
        self.statusbar.addWidget(self.status_label)
        self.statusbar.addPermanentWidget(self.progress_bar)

        self.connect_GUI_to_thread()
        self._init_data_control_singal()

    def _init_data_control_singal(self):
        # self.worker_thread.job_complete_signal.connect()
        pass

    

    def connect_GUI_to_thread(self):
        self.worker_thread.create_status_bar_signal.connect(self.make_status_progress)
        self.worker_thread.remove_status_bar_signal.connect(self.remove_status_progress)
        self.worker_thread.job_progress_signal.connect(self.write_status_progress)
    
    def connect_components(self):
        pass
    
    @pyqtSlot(uuid.UUID, signals.Event)
    def data_completed(self, ret_uuid, event):

        if signals.EventType.DATA_LOADED_FROM_META  in event:
            self.selected_data_chnaged_signal.emit(self.m_data_manager.get_selected_data())
        
        if signals.EventType.ALL_LANDMARK_DETECTED in event:
            self.selected_data_chnaged_signal.emit(self.m_data_manager.get_selected_data())

            

    @pyqtSlot(uuid.UUID, int, int, bool, str)
    def write_status_progress(self, uuid, cur_i, length, completed, msg):
        self.progress_bar.setValue(cur_i)
        self.status_label.setText(msg + "{}/{}".format(cur_i, length))
        
    @pyqtSlot(int, int)#init value, length
    def make_status_progress(self, start, length):
        self.progress_bar.setRange(start, length)
        self.progress_bar.setValue(start)
        self.progress_bar.setVisible(True)
    
    @pyqtSlot()
    def remove_status_progress(self):
        self.status_label.setText("")
        self.progress_bar.setVisible(False)



    
    def get_status_show_message(self):
        return self.statusbar.showMessage

    

    def closeEvent(self, event):
        self.worker_thread.terminate()
        print("terminated : ", self.worker_thread.isFinished())
        super(QMainWindow, self).closeEvent(event)


if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = MyApp()
   ex.show()
   sys.exit(app.exec_())