from PyQt5.QtWidgets import QFileDialog,QFileSystemModel,QAbstractItemView, QComboBox, QLabel, QApplication,QFormLayout,QSizePolicy,QHBoxLayout, QTextEdit,QPushButton, QVBoxLayout,QWidget, QDesktopWidget, QMainWindow, QGridLayout, QLabel, QLineEdit
from PyQt5 import QtGui, QtCore
import sys
import ctypes
import os.path as osp
import os 

class ImageWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.img_view = QLabel("test") 
      
        self.setSizePolicy(QSizePolicy.Expanding , QSizePolicy.Expanding )


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




class InspectorWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.data = dict()
        self.input_data = dict()
        self.input_data['root_directory'] = [QLineEdit(""), QPushButton("...")]
        self.input_data['save root location']  =[QLineEdit(""), QPushButton("...")]
        self.input_data['root_directory'][1].clicked.connect(self.find_root_directory)
        self.input_data['save root location'][1].clicked.connect(self.find_save_location)
        


        self.data['meta file'] = QLabel("")
        self.data['image_name'] = QLabel("")
        self.data['category'] = QLabel("")
        self.data['current image number']  = QLabel("?/?")
        self.detect_all_button = QPushButton("detect all lmk from whole images")
        self.detect_button = QPushButton("detect lmk")
        self.save_individual_button = QPushButton("save data")
        self.save_all_button = QPushButton("save all")

    def find_root_directory(self):
        #https://stackoverflow.com/questions/64530452/is-there-a-way-to-select-a-directory-in-pyqt-and-show-the-files-within
        dialog = QFileDialog(self, windowTitle='Select Root Directory')
        dialog.setDirectory(osp.abspath(os.curdir))
        dialog.setFileMode(dialog.Directory)
        dialog.setOptions(dialog.DontUseNativeDialog)        
        fileTypeCombo  = dialog.findChild(QComboBox, 'fileTypeCombo')

        if fileTypeCombo:
            fileTypeCombo.setVisible(False)
            dialog.setLabelText(dialog.FileType, '')
        
        if dialog.exec_():
           print(dialog.selectedFiles()[0])

    def find_save_location(self):
        pass

    def initUI(self):
        main_layout = QVBoxLayout()
        
        layout_info = QFormLayout()
        button_layout = QGridLayout()

        layout1 = QHBoxLayout()
        layout2 = QHBoxLayout()
        layout1.addWidget(self.input_data['root_directory'][0])
        layout1.addWidget(self.input_data['root_directory'][1])
        layout2.addWidget(self.input_data['save root location'][0])
        layout2.addWidget(self.input_data['save root location'][1])
        layout_info.addRow('root_directory', layout1)
        layout_info.addRow('save root location', layout2)

        for i, (key, item) in enumerate(self.data.items()):
            layout_info.addRow(key, item)

        button_layout.addWidget(self.detect_button,0, 0)
        button_layout.addWidget(self.save_individual_button,0, 1)
        button_layout.addWidget(self.save_all_button,1, 0, 1, 2)
        button_layout.addWidget(self.detect_all_button,2,0,1,2)
        main_layout.addLayout(layout_info)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
        self.show()

class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        width = ctypes.windll.user32.GetSystemMetrics(0)
        height = ctypes.windll.user32.GetSystemMetrics(1)
        self.setWindowTitle('landmark editor')
        self.resize(width*0.8, height*0.8)


        btn1 = ImageWidget()
        btn2 = InspectorWidget()
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
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())



if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = MyApp()
   ex.show()
   sys.exit(app.exec_())