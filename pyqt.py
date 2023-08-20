from PyQt5.QtWidgets import *
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal, QThread
import sys
import ctypes
import os.path as osp
import os 
import cv2

# import face_main_test as ff

class Data:
    def __init__(self, images_meta, lmk_meta):
        self.index = 0
        self.image_size = None 
        self.image_collection = None 
        self.is_load = False
    def is_loaded(self):
        return self.is_load

    def inc_index(self):
        if self.is_load:
            tmp = self.index + 1
            if tmp >= self.image_size:
                tmp = 0
            self.index = tmp 
    
    def dec_index(self):
        if self.is_load:
            tmp = self.index - 1
            if tmp < 0 :
                tmp = self.image_size - 1
            self.index = tmp 
    
    def __getitem__(self, idx):
        return self.image_collection[idx]
    
    def get_cur_index(self):
        return self.index

    def __len__(self):
        return self.image_size

    def detect_lmk(self, index = -1):
        if index == - 1: # detect current index
            pass
        else :
            pass

    def save(self, index):
        pass



class ImageWidget(QWidget):
    def __init__(self, program_data: Data):

        super().__init__()

        self.program_data = program_data

        self.img_view = QLabel("test") 
        self.img_view.setScaledContents(False)

        self.img_view.setAlignment(QtCore.Qt.AlignCenter)

        self.setSizePolicy(QSizePolicy.Ignored , QSizePolicy.Ignored )
        self.img_view.setStyleSheet("background-color: black; border: 1px solid black;")
        self.angle_ratio = 120
        self.scale_increase_size = 0.2
        self.reset_image_configuration()
        # self.img_view.wheelEvent = self.wheel


    def print_(self):
        size = self.img_view.pixmap().size()
        size = self.test.size()
        print(size)
    
    def reset_image_configuration(self):
        self.image_scale_factor = 1.0
        self.iamge_anchor = (0,0)
        self.anchor =(0,0)
    def mousePressEvent(self, e):
        print("clicked") 
    def mouseReleaseEvent(self, a0) -> None:
        print("released")
    def wheelEvent(self, event):
        self.image_scale_factor +=  self.scale_increase_size * event.angleDelta().y() / self.angle_ratio
        print(self.width(), self.height())
        pixmap = self.test
        myScaledPixmap = pixmap.scaled(self.image_scale_factor*pixmap.size(), QtCore.Qt.KeepAspectRatio)
        self.print_()
        self.img_view.setPixmap(myScaledPixmap)
        print(event.angleDelta().y())
        print(self.image_scale_factor)

    def getPos(self , event):
        print(event.button())
        x = event.pos().x()
        y = event.pos().y() 
        print(x, " , " , y)

    def initUI(self):
        print(self.width(), " ", self.height())
        self.test = QtGui.QPixmap()
        # test.load("./images/all_in_one/expression/GOPR0039.JPG")
        self.test.load("./test.JPG")
        test = self.test
        self.img_view.setPixmap(        test.scaled(self.width(), self.height()))
        self.img_view.setObjectName("image")
        self.img_view.mousePressEvent = self.getPos

        
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.img_view)
        self.setLayout(main_layout)

        self.show()
    

    def paint_event():
        pass

    def update_scale_increse_size(self, size):
        self.scale_increase_size = size

    def load_image_slot(self, index):
        self.program_data[index]

    def get_img_from_cv_img(self, cv_img):
        h,w,c = cv_img.shape
        ratio = 1.0
        canvas_h = self.height()
        canvas_w = self.width()
        if h > w:
            image_max_length = h
            ratio = canvas_h / image_max_length
        else:
            image_max_length = w
            ratio = canvas_w / image_max_length



        qImg = QtGui.QImage(cv_img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        self.img_view.setPixmap(pixmap)
        

    def listen_update_image(self):
        pass
class ImageWidget(QGraphicsView):
    pixmapClicked = pyqtSignal(QtCore.QPoint)
    def __init__(self, program_data: Data):

        self._scene = QGraphicsScene()
        super(ImageWidget, self).__init__(self._scene)

        self.program_data = program_data

        self.test = QtGui.QPixmap()
        self.test.load("./test.JPG")
        # self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        # self._scene.addPixmap()
        self.img_overlay = self._scene.addPixmap(self.test)
        
        
        
        self.pen = QtGui.QPen()
        self.brush = QtGui.QBrush(QtGui.QColor(0,0,255))
    
        
        self.reset_image_configuration()
        self.angle_ratio = 120
        self.scale_increase_size = 0.2
        
        
        self.right_mouse_pressed = False
        self.left_mouse_pressed = False
        
    def reset_image_configuration(self):
        self.image_scale_factor = 1.0
        pixel = 1
        self.circle_list  = []
        self.lines = []
        import random
        
        for i in range(68):
            test1 = random.randrange(0, 200)
            test2 = random.randint(0, 200)
            circle = self._scene.addEllipse(QtCore.QRectF(test1, test2, pixel, pixel),self.pen, self.brush)
            circle.test = i
            
            print(circle.mapToScene(circle.pos()).toPoint().x())
            print(circle.mapFromItem(circle.pos()).toPoint().x())
            print(circle.mapToItem(circle.pos()).toPoint().x())
            print(circle.mapToParent(circle.pos()).toPoint().x())
            print(circle.mapToScene(circle.pos()).toPoint().x())
            self.circle_list.append(circle)
        i = 0
        print(len(self.circle_list[:-1]), len(self.circle_list[1:]))
        for (t1,t2) in zip(self.circle_list[:-1], self.circle_list[1:]):
            e = t1.x()
            i+=1
            print(t1.x(), t1.y(), t2.x(), t2.y())
            line = QtCore.QLineF(t1.x(), t1.y(), t2.x(), t2.y())
            ll = self._scene.addLine(line, self.pen)
            self.lines.append(ll)     


        
    def initUI(self):
        print(self.width(), " ", self.height())
        self.test = QtGui.QPixmap()
        # test.load("./images/all_in_one/expression/GOPR0039.JPG")
        
        # main_layout = QVBoxLayout()
        # main_layout.addWidget(self._scene)
        # self.setLayout(main_layout)

        self.show()

    def mousePressEvent(self, event):
        
        vp = event.pos()
        if event.button() == QtCore.Qt.LeftButton:
            print(vp)
            if self.itemAt(vp) == self.img_overlay:
                sp = self.mapToScene(vp)
                lp = self.img_overlay.mapFromScene(sp).toPoint()
                print("test")
                print(lp)
                self.pixmapClicked.emit(lp)
            elif self.itemAt(vp) in self.circle_list :
                print(self.itemAt(vp).test)
                print(self.itemAt(vp).isEnabled())
            else : # this case line 
                pass
                
        elif event.button() == QtCore.Qt.RightButton:
                self.loc = vp
                self.right_mouse_pressed = True

        super(ImageWidget, self).mousePressEvent(event)
        
    def mouseReleaseEvent(self, event) -> None:
        print("release")
        vp = event.pos()
        if event.button() == QtCore.Qt.LeftButton:
            vp = event.pos()
            # print(vp)
            if self.itemAt(vp) == self.img_overlay:
                sp = self.mapToScene(vp)
                lp = self.img_overlay.mapFromScene(sp).toPoint()
                print(lp)
                self.pixmapClicked.emit(lp)
        elif event.button() == QtCore.Qt.RightButton:
            self.right_mouse_pressed = False
            print('right')
        return super().mouseReleaseEvent(event)
    
    def mouseMoveEvent(self, e):  # e ; QMouseEvent
        print(e.button())
        print(QtCore.Qt.RightButton)
        if self.right_mouse_pressed:
            transform = self.transform()
            delta_x = e.pos().x() - self.loc.x()
            delta_y = e.pos().y() - self.loc.y()
            delta_x = delta_x/transform.m11()
            delta_y = delta_y/transform.m22()
            self.setSceneRect(self.sceneRect().translated(-delta_x, -delta_y))

            self.loc = e.pos()
            print("dy ", delta_x, " ", delta_y)
            ratio = 10
            self.transform()
            self.translate(delta_x*10, delta_y*10)
            
            
        # print('x y (%d %d)' % (e.x(), e.y()))

    def wheelEvent(self, e):  # e ; QWheelEvent
        ratio = e.angleDelta().y() / self.angle_ratio
        scale = 1.0
        scale +=  self.scale_increase_size * (ratio)
        # print(ratio)
        # print(scale)
        self.scale(scale, scale)

class InspectorSignalCollection(QObject):
    InspectorIndexSignal = pyqtSignal()
    InspectorLmkDetectSignal = pyqtSignal(int)
    InspectorSaveSignal = pyqtSignal(int)
    def call_cur_index(self):
        print("call_cur_index")
        self.InspectorIndexSignal.emit()
    def call_inspector_detect_signal(self, index):
        print("call_inspector_detect_signal", index)
        self.InspectorLmkDetectSignal.emit(index)
    def call_save_signal(self, index):
        print("call_save_signal", index)
        self.InspectorSaveSignal.emit(index)
        

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



        self.signal = InspectorSignalCollection()
        self.init_button_signal()

    def init_button_signal(self):
        
        def function_wrapping(*funcs):
            def wrapper():
                for f in funcs:
                    if isinstance(f, (list, tuple)):
                        f[0](f[1:])
                    else:
                        f()
            return wrapper
        
        prev_index_function = function_wrapping(self.program_data.dec_index, self.signal.call_cur_index)
        next_index_function = function_wrapping(self.program_data.inc_index, self.signal.call_cur_index)
        self.prev.clicked.connect( prev_index_function )
        self.next.clicked.connect( next_index_function )


        detect_function = function_wrapping(self.program_data.detect_lmk, [self.signal.call_inspector_detect_signal, self.program_data.get_cur_index()])

        def detect_all_lmk_function(): 
            for i in range(len(self.program_data)):
                self.program_data.detect_lmk(i)
            self.signal.call_inspector_detect_signal(i)
        detect_all_function = function_wrapping(detect_all_lmk_function, )
        self.detect_all_button.clicked.connect(detect_function)
        self.detect_button.clicked.connect(detect_all_function)


        def save_function():
            self.program_data.save()
            self.signal.call_save_signal(self.program_data.get_cur_index())
        def save_all_function():
            for i in range(len(self.program_data)):
                self.program_data.save(i)
                self.signal.call_save_signal(i)

        self.save_individual_button.clicked.connect(save_function)
        self.save_all_button.clicked.connect(save_all_function)

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
        self.worker_thread = Worker()


        self.initUI()

    def initUI(self):
        width = ctypes.windll.user32.GetSystemMetrics(0)
        height = ctypes.windll.user32.GetSystemMetrics(1)
        self.setWindowTitle('landmark editor')
        self.resize(width*0.8, height*0.8)

        data =Data(None, None)
        btn1 = ImageWidget(data)
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