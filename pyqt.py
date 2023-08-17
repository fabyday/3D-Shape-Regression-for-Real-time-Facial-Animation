from PyQt5.QtWidgets import QLabel, QApplication,QFormLayout,QSizePolicy,QHBoxLayout, QTextEdit,QPushButton, QVBoxLayout,QWidget, QDesktopWidget, QMainWindow, QGridLayout, QLabel, QLineEdit
from PyQt5 import QtGui
import sys
import ctypes

class ImageWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.img_view = QPushButton("test") 
        self.initUI()
        


    def initUI(self):
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
        self.data['root_directory'] = QLabel("")
        self.data['meta file'] = QLabel("")
        self.data['image_name'] = QLabel("")
        self.data['category'] = QLabel("")
        self.data['save root location']  =QLineEdit("")
        self.data['current image number']  = QLabel("?/?")

        self.detect_all_button = QPushButton("detect all lmk from whole images")
        self.detect_button = QPushButton("detect lmk")
        self.save_individual_button = QPushButton("save data")
        self.save_all_button = QPushButton("save all")


        

        self.initUI()
    def initUI(self):
        main_layout = QVBoxLayout()

        layout_info = QFormLayout()
        button_layout = QGridLayout()
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

        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.addWidget(btn1)
        layout.addStretch()

        layout.addWidget(btn2)
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