import typing
from PyQt5.QtWidgets import *
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QImage
from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal, QThread
import sys
import ctypes
import os.path as osp
import os 
import cv2
import yaml
import face_main_test as ff

class Data:
    def __init__(self, images_meta, lmk_meta):
        self.index = 0
        self.image_size = None 
        self.image_collection = None 
        self.load(images_meta, lmk_meta)
        self.is_load = False
        self.save_location = ""

    def set_save_location(self, path):
        self.save_location = path



    def load(self, images_meta, lmk_meta):
        if images_meta : 
            self.image_collection =  ff.ImageCollection(image_dir=images_meta, lmk_meta_file=lmk_meta, lazy_load=True)
            self.image_size = len(self.image_collection)
        self.is_load = True

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

    def save(self, index):
        pass
    
    def save_all(self):
        pass

    def changed_lmk_listner(self, image_index, item):
        print("test")
        try:
            lmk_num = item.num
            x = item.pos().x()
            y = item.pos().y()
            self.image_collection[image_index].set_lmk(lmk_num, x,y)
        except:
            pass
    

        


class Worker(QThread):
    image_load_finished = pyqtSignal(int)
    lmk_load_finished = pyqtSignal(int)
    job_finished = pyqtSignal(int, int, bool)
    def __init__(self, parent, program_data) -> None:
        super().__init__(parent)
        self.function_list = []
        self.program_data = program_data
        self.pp = parent
    def run(self):
        while True:
            if len(self.function_list):
                f = self.function_list.pop(0)
                f()

    def load_detector(self, path = None):
        write, fin = self.pp.make_progress_bar_at_status_bar(0, 1)
        def wrapper():
            try:
                write(0, "loading start")
                self.program_data.image_collection.load_predictor(path)
                write(0, "loading end")
            except:
                lab = self.pp.get_status_show_message()
                lab("this funtionality is not allowed. because meta file is not loaded", 2000)
            finally:
                fin()
        
        self.function_list.append(wrapper)

    


    def save_data(self, id, index_list, callback= None ):
        total_size = len(index_list)
        cancel_flag = False
        def cancel_f():
            nonlocal cancel_flag
            cancel_flag = True
        def wrapper():
            finished_index_list = []
            write, fin = self.pp.make_progress_bar_at_status_bar(0, len(index_list))
            for i, index in enumerate(index_list):
                if not cancel_flag:
                    self.program_data[index].redetect_flag = True
                    self.program_data[index].calc_lankmark_from_dlib()
                    finished_index_list.append(i)
                    if callback != None :
                        callback(i, total_size)
                    write(i + 1, self.program_data[index].img_name + " finished.")
                else:
                    fin()
                    self.job_finished.emit(id,finished_index_list, False)
                    return
            fin()
            self.job_finished.emit(id,finished_index_list, True)

        self.function_list.append(wrapper)
        return cancel_f

    def do_detect_lmk(self, id, index_list, callback = None):
        total_size = len(index_list)
        cancel_flag = False
        def cancel_f():
            nonlocal cancel_flag
            cancel_flag = True
        def wrapper():
            finished_index_list = []
            write, fin = self.pp.make_progress_bar_at_status_bar(0, len(index_list))
            for i, index in enumerate(index_list):
                if not cancel_flag:
                    self.program_data[index].redetect_flag = True
                    self.program_data[index].calc_lankmark_from_dlib()
                    finished_index_list.append(i)
                    if callback != None :
                        callback(i, total_size)
                    write(i + 1, self.program_data[index].img_name + " finished.")
                else:
                    fin()
                    self.job_finished.emit(id,finished_index_list, False)
                    return
            fin()
            self.job_finished.emit(id,finished_index_list, True)
        self.function_list.append(wrapper)
        return cancel_f

    def do_load_image(self, id, index_list, callback = None):
        total_size = len(index_list)
        cancel_flag = False
        def cancel_f():
            nonlocal cancel_flag
            cancel_flag = True
        def wrapper():
            nonlocal cancel_flag
            finished_index_list = []
            for i, index in enumerate(index_list):
                if not cancel_flag:
                    self.program_data[index].load()
                    finished_index_list.append(i)
                    if callback != None:
                        callback(index, total_size)
                else:
                    self.job_finished.emit(id, finished_index_list, False)
            self.job_finished.emit(id, finished_index_list, True)
        self.function_list.append(wrapper)
        return cancel_f




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
    lmk_data_changed_signal = pyqtSignal(int, QGraphicsEllipseItem)
    def __init__(self, program_data: Data, worker : Worker):

        self._scene = QGraphicsScene()
        super(ImageWidget, self).__init__(self._scene)
        self.worker = worker

        self.worker.lmk_load_finished.connect(self.reload_image)
        self.worker.lmk_load_finished.connect(self.reload_lmk_to_view)

        self.program_data = program_data

        self.test = QtGui.QPixmap()
        self.test.load("./test.JPG")
        # self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        # self._scene.addPixmap()
        self.img_overlay = self._scene.addPixmap(self.test)
        
        
        # draw color
        self.pen = QtGui.QPen()
        self.brush = QtGui.QBrush(QtGui.QColor(0,0,0))
        self.point = QtGui.QBrush(QtGui.QColor(0,0,0))

        self.outer_upper_side_brush = QtGui.QBrush(QtGui.QColor(255,0,0))
        self.inner_upper_side_brush = QtGui.QBrush(QtGui.QColor(0,255,255))
        self.inner_lower_side_brush = QtGui.QBrush(QtGui.QColor(255,255,0))
        self.outer_lower_side_brush = QtGui.QBrush(QtGui.QColor(0,0,255))

        self.reset_image_configuration()
        self.angle_ratio = 120
        self.scale_increase_size = 0.2
        
        self.selected_pts = None 
        self.right_mouse_pressed = False
        self.left_mouse_pressed = False


        self.index = -1
    

    def set_image_index(self, index):
        self.index = index
    #qt callback function
    def reload_image(self):
        index = self.index
        image = self.program_data[index]
        if len(image.img.shape)<3:
            frame = cv2.cvtColor(image.img, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(image.img, cv2.COLOR_BGR2RGB)

        h, w = image.img.shape[:2]
        bytesPerLine = 3 * w

        qimage = QImage(frame.data, w, h, bytesPerLine, QImage.Format.Format_RGB888)
        test = QtGui.QPixmap()
        test.fromImage(qimage)
        # self.img_overlay.pixmap().convertFromImage(qimage)
        self.img_overlay.setPixmap(test)
        self.reload_lmk_to_view()
    
    # this method will be called by callback and image load function
    def reload_lmk_to_view(self):
        index = self.index
        image = self.program_data[index]
        lmks = image.lmk
        for lmk, circle in zip(lmks, self.circle_list):
            x = lmk[0]
            y = lmk[1]
            circle.setPos(x, y)
            circle.setVisible(True)
        
        for i, (c1,c2) in enumerate(zip(self.circle_list[:-1], self.circle_list[1:])):
            self.lines[i].setLine(c1.pos().x(), c1.pos().y(), c2.pos().x(), c2.pos().y())
            self.lines[i].setVisible(True)

    def reset_image_configuration(self):
        self.image_scale_factor = 1.0
        pixel = 1
        self.circle_list  = []
        self.lines = []
        import random
        
        for i in range(68):
            test1 = random.randrange(0, 200)
            test2 = random.randint(0, 200)
            circle = self._scene.addEllipse(QtCore.QRectF(-pixel/2, -pixel/2, pixel/2, pixel/2),self.pen, self.brush)
            circle.setPos(test1, test2)
            circle.setZValue(1)
            circle.num = i
            circle.connected_line = set()
            self.circle_list.append(circle)

        for i, (t1,t2) in enumerate(zip(self.circle_list[:-1], self.circle_list[1:])):
            e = t1.x()
            line = QtCore.QLineF(t1.x(), t1.y(), t2.x(), t2.y())
            ll = self._scene.addLine(line, self.pen)
            t1.connected_line.add(ll)
            t2.connected_line.add(ll)
            ll.connected_pts = []
            ll.connected_pts.append(t1)
            ll.connected_pts.append(t2)
            self.lines.append(ll)     

    def circle_line_edit(self, changed_pts):
        line_set = changed_pts.connected_line

        for ll in line_set:
            c1 = ll.connected_pts[0]
            c2 = ll.connected_pts[1]
            ll.setLine(c1.pos().x(), c1.pos().y(), c2.pos().x(), c2.pos().y())

    def circle_line_visible(self, flag:bool):
        for circle in self.circle_list:
            circle.setVisible(flag)
        for line in self.lines:
            line.setVisible(flag)
        
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
                # self.pixmapClicked.emit(lp)
            elif self.itemAt(vp) in self.circle_list :
                self.selected_pts = self.itemAt(vp)
            else : # this case line 
                sp = self.mapToScene(vp)
                lp = self.img_overlay.mapFromScene(sp).toPoint()
                print(lp)


                
        elif event.button() == QtCore.Qt.RightButton:
                self.loc = vp
                self.right_mouse_pressed = True

        super(ImageWidget, self).mousePressEvent(event)
        
    def mouseReleaseEvent(self, event) -> None:
        print("release")
        vp = event.pos()
        if event.button() == QtCore.Qt.LeftButton:
            self.selected_pts = None
            vp = event.pos()
            # print(vp)
            if self.itemAt(vp) == self.img_overlay:
                sp = self.mapToScene(vp)
                lp = self.img_overlay.mapFromScene(sp).toPoint()
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
            ratio = 10
            self.transform()
            self.translate(delta_x*10, delta_y*10)
        elif e.buttons() == QtCore.Qt.LeftButton:
            # print(e.pos().x(), e.pos().y())
            # print(lp.x(), lp.y())
            if self.selected_pts :
                sp = self.mapToScene(e.pos())
                lp = self.img_overlay.mapFromScene(sp).toPoint()
                self.selected_pts.setPos(lp.x(), lp.y())
                self.circle_line_edit(self.selected_pts)
                self.lmk_data_changed_signal.emit(self.index, self.selected_pts)
            
        # print('x y (%d %d)' % (e.x(), e.y()))

    def wheelEvent(self, e):  # e ; QWheelEvent
        ratio = e.angleDelta().y() / self.angle_ratio
        scale = 1.0
        scale +=  self.scale_increase_size * (ratio)
        # print(ratio)
        # print(scale)
        self.scale(scale, scale)

class InspectorSignalCollection(QObject):
    InspectorIndexSignal = pyqtSignal(int)
    InspectorLmkDetectSignal = pyqtSignal(int)
    InspectorSaveSignal = pyqtSignal(int)
    
        

class InspectorWidget(QWidget):
    def __init__(self, p, program_data : Data, worker : Worker):

        super().__init__()
        self.pp = p
        self.program_data = program_data
        self.worker = worker
        self.data = dict()
        self.input_data = dict()
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.input_data['root_directory'] = [QLineEdit(""), QPushButton("...")]
        self.input_data['save root location']  =[QLineEdit(""), QPushButton("...")]
        self.input_data['root_directory'][0].setReadOnly(True)
        self.input_data['save root location'][0].setReadOnly(True)
        
        self.jobs = dict()
        self.current_job_item = (None, None)


        

        def check_and_load_meta_file(name):
            self.input_data['root_directory'][0].setText(name)
            if osp.exists(osp.join(name, "meta.yaml")):
                self.program_data.load(name, "ict_lmk_info.yaml")
                self.data['meta file'].setText("meta.yaml")
                self.upate_ui()
                self.worker.load_detector()
            else:
                self.data["meta file"].setText("Not Found.")

     
        self.input_data['root_directory'][1].clicked.connect(self.open_and_find_directory(check_and_load_meta_file))
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
    def upate_ui(self):
        lab = self.pp.get_status_show_message()

        try :
            self.image_info[-1].setText(str(len(self.program_data)))
            self.image_info[1].setText(str(self.program_data.get_cur_index() + 1))
            item = self.program_data[self.program_data.get_cur_index()]
            self.data['image_name'].setText(item.img_name)
            self.data['category'].setText(item.category)
            onlyInt = QIntValidator()
            self.setRange(1, len(self.program_data))
            self.image_info[1].setValidator(onlyInt)
        except:
            lab("this funtionality is not allowed. because meta file is not loaded", 2000)

    def init_button_signal(self):
    
        import time

        def update_when_finished_cur_image(i, total_size):
                print("tal")
                print(i)
                print(self.program_data.get_cur_index())
                if i == self.program_data.get_cur_index():
                    print("test")
                    self.signal.InspectorIndexSignal.emit(self.program_data.get_cur_index())
           
        def prev():
            lab = self.pp.get_status_show_message()

            try:
                j_id = int(time.time())
                self.program_data.dec_index()
                cancel_f = self.worker.do_load_image(j_id, [self.program_data.get_cur_index()],  update_when_finished_cur_image)
                self.current_job_item = (j_id , cancel_f)


                self.upate_ui()
            except:
                lab("this funtionality is not allowed. because meta file is not loaded", 2000)


            
        def next():
            lab = self.pp.get_status_show_message()

            try:
                j_id = int(time.time())
                self.program_data.inc_index()
                cancel_f = self.worker.do_load_image(j_id, [self.program_data.get_cur_index()],  update_when_finished_cur_image)
                self.current_job_item = (j_id , cancel_f)
                self.upate_ui()
                self.jobs[j_id] = False

            except:
                lab("this funtionality is not allowed. because meta file is not loaded", 2000)


        self.prev.clicked.connect( prev )
        self.next.clicked.connect( next )



        def detect(index_list = None):

            def update_when_finished_cur_image(i, total_size):
                if i == self.program_data.get_cur_index():
                    self.signal.InspectorLmkDetectSignal.emit(self.program_data.get_cur_index())
           
            lab = self.pp.get_status_show_message()
            try:
                if index_list == None:
                    index_list = [self.program_data.get_cur_index()]
                

                if self.current_job_item == None :
                    j_id = int(time.time())
                    cancel_f = self.worker.do_detect_lmk(j_id, index_list, update_when_finished_cur_image)
                    self.current_job_item = (j_id , cancel_f)
                    self.jobs[j_id] =False
                else:
                    self.current_job_item[1]() # cancel function
                    j_id = int(time.time())
                    cancel_f = self.worker.do_detect_lmk(j_id, index_list, update_when_finished_cur_image)
                    self.current_job_item = (j_id , cancel_f)
                    self.jobs[j_id] = False
            except:
                lab("this funtionality is not allowed. because meta file is not loaded", 2000)
                
        def detect_all():
            lab = self.pp.get_status_show_message()

            try:
                detect(list(range(len(self.program_data))))
            except:
                lab("this funtionality is not allowed. because meta file is not loaded.", 2000)      
        def detect_finished_function(j_id, index_list, flag):
            # emitted by thread
            if flag == False : #when canceled
                delete_f = self.jobs.pop(j_id)
            if flag == True :
                self.current_job_item = None 
                delete_f = self.jobs.pop(j_id)


        self.worker.job_finished.connect(detect_finished_function)


        self.detect_all_button.clicked.connect(detect_all)
        self.detect_button.clicked.connect(detect)


        def save_function():
            lab = self.pp.get_status_show_message()
            try:
                self.program_data.save(self.program_data.get_cur_index())
                self.signal.call_save_signal(self.program_data.get_cur_index())
            except:
                lab("this funtionality is not allowed. because meta file is not loaded.", 2000)      

        def save_all_function():
            lab = self.pp.get_status_show_message()
            try:
                for i in range(len(self.program_data)):
                    self.program_data.save(i)
                    self.signal.call_save_signal(i)
            except:
                lab("this funtionality is not allowed. because meta file is not loaded.", 2000)      

            

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

        self.data = Data(None, None)
        self.worker_thread = Worker(self, self.data)
        self.worker_thread.start()

        self.initUI()
        self.connect_widgets_functionality()
        # self.worker_thread.load_detector()
    def initUI(self):
        width = ctypes.windll.user32.GetSystemMetrics(0)
        height = ctypes.windll.user32.GetSystemMetrics(1)
        self.setWindowTitle('landmark editor')
        self.resize(width*0.8, height*0.8)

        imagewidget = ImageWidget(self.data, self.worker_thread)
        inspectorwidget = InspectorWidget(self, self.data, self.worker_thread)
        # btn1.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        # btn1.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        widget = QWidget()

        layout = QHBoxLayout(widget)
        layout.addWidget(imagewidget, 7)
        layout.addWidget(inspectorwidget, 3)
        imagewidget.initUI()
        inspectorwidget.initUI()
        self.setCentralWidget(widget)

        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_label = QLabel()
        self.statusbar.addWidget(self.status_label)
        self.statusbar.addPermanentWidget(self.progress_bar)
        self.imagewidget = imagewidget
        self.inspectorwidget = inspectorwidget



        

    def make_progress_bar_at_status_bar(self, start, end):
        self.progress_bar.setRange(start, end)
        self.progress_bar.setValue(start)
        self.progress_bar.setVisible(True)
        def write(i, message=""):
            self.progress_bar.setValue(i)
            self.status_label.setText(message)
        def finished():
            self.progress_bar.setVisible(False)

        return write, finished

    def get_status_show_message(self):

        return self.statusbar.showMessage

    def connect_widgets_functionality(self):
        def wrapper(i):
            self.imagewidget.set_image_index(i)
            self.imagewidget.reload_image()
            self.imagewidget.reload_lmk_to_view()
        
        self.imagewidget.lmk_data_changed_signal.connect(self.data.changed_lmk_listner)
        self.inspectorwidget.signal.InspectorLmkDetectSignal.connect(self.imagewidget.reload_lmk_to_view)
        self.inspectorwidget.signal.InspectorIndexSignal.connect(wrapper)

    def center(self):
        # qr = self.frameGeometry()
        # cp = QDesktopWidget().availableGeometry().center()
        # qr.moveCenter(cp)
        # self.move(qr.topLeft())
        pass
    

    def closeEvent(self, event):
        self.worker_thread.terminate()
        print("terminated : ", self.worker_thread.isFinished())
        # here you can terminate your threads and do other stuff
        # and afterwards call the closeEvent of the super-class
        super(QMainWindow, self).closeEvent(event)


if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = MyApp()
   ex.show()
   sys.exit(app.exec_())