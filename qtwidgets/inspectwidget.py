from PyQt5.QtWidgets import *
import time
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QImage
from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal, QThread, QMutex
import os 
import os.path as osp
import logger 
import uuid 
import typing
from PyQt5.QtWidgets import QWidget, QMessageBox



root_logger = logger.root_logger
inspector_logger = root_logger.getChild("inspector panel")
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

class InspectorWidget(QWidget):

    detector_loaded = pyqtSignal(bool)
    load_landmark_meta = pyqtSignal()
    selected_data_changed = pyqtSignal(uuid.UUID)

    def __init__(self, ctx, parent ):
        super().__init__()
        self.m_parent = parent
        self.m_ctx = ctx
        self.m_root_dir = [QLineEdit(""), QPushButton("...")]
        self.m_save_root_loc = [QLineEdit(""), QPushButton("...")]

        
        project_root_path = osp.abspath(osp.join(osp.dirname(__file__), ".."))
        landmark_default_path = osp.join(project_root_path, "ict_lmk_info.yaml")
        self.m_ctx.load_landmark_meta(landmark_default_path)
        self.load_landmark_meta.emit()
        self.m_landmark_meta = [QLineEdit(landmark_default_path), QPushButton("...")]
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

        self.handler = logger.QTConsoleHandler(self)

        logger.root_logger.addHandler(self.handler)        

        self.console_text = QtWidgets.QPlainTextEdit(self)
        self.console_text.setReadOnly(True)
        self.handler.appendPlainText.connect(self.console_text.appendPlainText)
        console_text_layout = QVBoxLayout()
        console_text_layout.addWidget(self.console_text)
        main_layout.addLayout(console_text_layout)


        


        self.setLayout(main_layout)
        self.init_connection()
        self.show() 
        inspector_logger.debug("inspector logger inspect ui initialization was successed.")
        try :
            self.m_ctx.get_landmark_structure_meta()
            inspector_logger.debug("default ict landmark structure was loaded")
        except:
            inspector_logger.debug("default ict landmark structure was not loaded...")

    
    def _update_inspector_to_current_target(self, index_o_uuid):
        item = self.m_ctx.get_meta()[index_o_uuid]
        self.m_category.setText(item.m_parent_category.category_name)
        self.m_image_name.setText(item.name)
        self.m_image_info[1].setText(str(self.m_ctx.get_selected_data_index()))
        self.m_image_info[-1].setText(str(len(self.m_ctx)))

    # event emitter
    def next(self):
        prev_index = self.m_ctx.get_selected_data_index()
        print(len(self.m_ctx))
        index = 0 if prev_index + 1 >= len(self.m_ctx) else prev_index + 1
        inspector_logger.debug("action next : [%s move to %s] [%s / %s]", str(prev_index), str(index), str(index),str(len(self.m_ctx)))
        key = self.m_ctx.set_selected_data_from_index(index)
        self._update_inspector_to_current_target(key)
        
        self.selected_data_changed.emit(key)

    def prev(self):
        prev_index = self.m_ctx.get_selected_data_index()
        index = len(self.m_ctx) - 1 if prev_index -1 < 0 else prev_index - 1

        inspector_logger.debug("action prev : [%s move to %s] [%s / %s]", str(prev_index), str(index),str(index), str(len(self.m_ctx)))
        key = self.m_ctx.set_selected_data_from_index(index)
        self._update_inspector_to_current_target(key)
        self.selected_data_changed.emit(key) 

    def detect(self):
        pass

    def save_data(self):
        inspector_logger.debug("action : save_data")
        inspector_logger.debug("save location :%s", self.m_save_root_loc[0].text())
        if self.m_save_root_loc[0].text() == "":
            inspector_logger.warn("check save root location. it was invalid.")
            reply = QMessageBox.information(self, 'warning', 'please setup save root location.')
        else:
            self.m_ctx.save_data(self.m_save_root_loc.text())  

    def save_data_all(self):
        inspector_logger.debug("action : save_data_all")
        inspector_logger.debug("save location :%s", self.m_save_root_loc[0].text())
        if self.m_save_root_loc[0].text() == "":
            inspector_logger.warn("check save root location. it was invalid.")
            reply = QMessageBox.information(self, 'warning', 'please setup save root location.')
        else:
            self.m_ctx.save_data(self.m_save_root_loc[0].text())

    def detect_all_lmk_from_entire_images(self):
        inspector_logger.debug("action : detect_landmark entire images")

        self.m_ctx.detect_all_landmark()
        # key = self.m_ctx.get_selected_data()
        # self.selected_data_changed.emit(key)    

    def open_root_dir_and_load(self, pth_name):
        try : 
            self.root_dir_widget = pth_name
            self.m_ctx.load_data_from_meta(pth_name)
        except  FileNotFoundError:
            return # if path was not founded. pass the loading meta.
        

    def open_save_loc(self, pth_name):
        try : 
            inspector_logger.debug("set save location : %s", pth_name)
            self.m_save_root_loc[0].setText(pth_name)
        except FileNotFoundError :
             self.m_save_root_loc[0].setText("")
    

    def open_meta_file(self, pth):
        self.m_ctx.load_data_from_meta(pth)
        
    #################################################################################################

    def init_connection(self):
        # self.program_data.save_location = self.input_data['save root location'][0].text()
        # self.root_dir_widget.clicked.connect(open_and_find_directory(self, lambda name : self.m_root_dir[0].setText(name)))
        self.root_dir_widget.clicked.connect(open_and_find_directory(self, self.open_root_dir_and_load))
        self.save_root_loc.clicked.connect(open_and_find_directory(self, self.open_save_loc))
        # self.m_landmark_meta[1].clicked.connect(open_and_find_meta(self, self.open_lmk_meta_file))
        
        self.m_prev_btn.clicked.connect(self.prev)
        self.m_next_btn.clicked.connect(self.next)

        # self.m_detect_button.clicked.connect()
        self.m_detect_all_button.clicked.connect(self.detect_all_lmk_from_entire_images)

        # self.m_save_individual_button.clicked.connect()
        self.m_save_all_button.clicked.connect(self.save_data_all)


        self.m_parent.selected_data_changed_signal.connect(self._update_inspector_to_current_target)

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
    
    