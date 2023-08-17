from PyQt5.QtWidgets import *
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal, QThread

import sys
import ctypes
import os.path as osp
import os 





class ImageWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.img_view = QLabel("test") 
      
        self.setSizePolicy(QSizePolicy.Expanding , QSizePolicy.Expanding )
        self.img_view.setStyleSheet("background-color: black; border: 1px solid black;")

    def getPos(self , event):
        x = event.pos().x()
        y = event.pos().y() 
        print(x, " , " , y)

    def initUI(self):
        print(self.width(), " ", self.height())
        test = QtGui.QPixmap()
        test.load("./images/all_in_one/expression/GOPR0039.JPG")
        self.img_view.setPixmap(        test.scaled(self.width(), self.height()))
        self.img_view.setObjectName("image")
        self.img_view.mousePressEvent = self.getPos

        
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.img_view)
        self.setLayout(main_layout)

        self.show()
    

    def get_img_from_cv_img(self, cv_img):
        h,w,c = cv_img.shape
        qImg = QtGui.QImage(cv_img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        self.img_view.setPixmap(pixmap)


    def listen_update_image(self):
        pass

class Data:
    def __init__(self, images_meta, lmk_meta):
        self.index = 0
        self.image_size = None 
        self.image_collection = None 
    
    def inc_index(self):
    
        tmp = self.index + 1
        if tmp >= self.image_size:
            tmp = 0
        self.index = tmp 
    
    def dec_index(self):
        tmp = self.index - 1
        if tmp < 0 :
            tmp = self.image_size - 1
        self.index = tmp 
    
    def index(self, idx):
        return self.image_collection[idx]




class InspectorIndexSignal(QObject):
    signal = pyqtSignal()

    def call(self):
        self.signal.emit()


class InspectorLmkDetectSignal(QObject):
    signal = pyqtSignal()

    def call(self):
        self.signal.emit()

class Worker(QThread):
    finished = pyqtSignal(int)



    def do_wrok(self):
        pass



class InspectorWidget(QWidget):
    def __init__(self, program_data : Data):

        super().__init__()


        self.program_data = program_data

        self.data = dict()
        self.input_data = dict()
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.input_data['root_directory'] = [QLineEdit(""), QPushButton("...")]
        self.input_data['save root location']  =[QLineEdit(""), QPushButton("...")]
        self.input_data['root_directory'][0].setReadOnly(True)
        self.input_data['save root location'][0].setReadOnly(True)

        def check_meta_file(name):
            self.input_data['root_directory'][0].setText(name)
            if osp.exists(osp.join(name, "meta.yaml")):
                self.data["meta file"].setText("meta.yaml")
            self.data["meta file"].setText("Not Found.")



        self.input_data['root_directory'][1].clicked.connect(self.open_and_find_directory(check_meta_file))
        self.input_data['save root location'][1].clicked.connect(self.open_and_find_directory(lambda name : self.input_data['save root location'][0].setText(name)))
        


        self.data['meta file'] = QLineEdit("Not Found.")
        self.data['meta file'].setReadOnly(True)
        self.data['image_name'] = QLineEdit("Not Found.")
        self.data['image_name'].setReadOnly(True)
        self.data['category'] = QLineEdit("Not Found.")
        self.data['category'].setReadOnly(True)

        self.image_info  = [QLabel("image number"), QLineEdit("?"), QLabel("/"), QLabel("?")]
        self.prev = QPushButton("prev")
        self.next = QPushButton("next")

        self.prev.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.next.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        
        self.detect_all_button = QPushButton("detect all lmk from whole images")
        self.detect_button = QPushButton("detect lmk")
        self.save_individual_button = QPushButton("save data")
        self.save_all_button = QPushButton("save all")



        self.signal = InspectorIndexSignal()
        self.init_button_signal()

    def init_button_signal(self):
        
        def index_function(func):
            def wrapper():
                func()
                self.signal.signal.call()

            return wrapper
        self.prev.clicked.connect( index_function(self.program_data.inc_index ))
        self.next.clicked.connect(index_function(self.program_data.dec_index ))




        self.detect_all_button.clicked.connect()
        self.detect_button.clicked.connect()

        self.save_individual_button.clicked.connect()
        self.save_all_button.clicked.connect()

    def setup_image_number(self, start_index, size):
        self.image_info[1].setText(str(start_index + 1))
        self.image_info[2].setText(size)


    def open_and_find_directory(self, callback):
        # callback input is file name
        #https://stackoverflow.com/questions/64530452/is-there-a-way-to-select-a-directory-in-pyqt-and-show-the-files-within
        def wrapper():
            dialog = QFileDialog(self, windowTitle='Select Root Directory')
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
    
    def find_save_location(self):
        pass

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(-1,0,-1,-1)
        main_layout.setAlignment(QtCore.Qt.AlignTop)
        layout_info = QFormLayout()
        button_layout = QGridLayout()

        layout1 = QHBoxLayout()
        layout2 = QHBoxLayout()
        layout1.addWidget(self.input_data['root_directory'][0])
        layout1.addWidget(self.input_data['root_directory'][1])

        layout2.addWidget(self.input_data['save root location'][0])
        layout2.addWidget(self.input_data['save root location'][1])
        
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

class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.data = None


        self.initUI()

    def initUI(self):
        width = ctypes.windll.user32.GetSystemMetrics(0)
        height = ctypes.windll.user32.GetSystemMetrics(1)
        self.setWindowTitle('landmark editor')
        self.resize(width*0.8, height*0.8)

        data =Data(None, None)
        btn1 = ImageWidget()
        btn2 = InspectorWidget(data)
        # btn1.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        # btn1.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        widget = QWidget()

        layout = QHBoxLayout(widget)
        layout.addWidget(btn1, 7)
        layout.addWidget(btn2, 3)
        btn1.initUI()
        btn2.initUI()
        self.setCentralWidget(widget)


    def center(self):
        # qr = self.frameGeometry()
        # cp = QDesktopWidget().availableGeometry().center()
        # qr.moveCenter(cp)
        # self.move(qr.topLeft())
        pass



if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = MyApp()
   ex.show()
   sys.exit(app.exec_())