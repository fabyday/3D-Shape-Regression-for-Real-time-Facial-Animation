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
import face_main_test as ff
import debugpy

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

    def get_detector_and_predictor(self):
        return self.image_collection.get_detector_and_predictor()

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


    def _init_meta(self):
        self.meta = {"meta" : dict()}
        self.meta['meta']['images_name'] = dict()

        

    def _add_meta(self, img_object):
        
        category = self.meta['meta']['images_name'].get(img_object.category, None)
        if category == None :
            category = self.meta['meta']['images_name'][img_object.category] = []
        img_meta = dict()
        img_meta['name'] = img_object.img_name
        img_meta['landmark'] = self.save_lmk_data(self.save_location, img_object)
        category.append(img_meta)
        
    def _meta_save(self):
        with open(osp.join(self.save_location, "meta.yaml"), "w") as fp:
            yaml.dump(self.meta, fp)

    def save_lmk_data(self, path, image_data):
        lmk = image_data.lmk
        img_name = image_data.img_name
        if not osp.exists(path):
            os.makedirs(path)
        saved_path = osp.join(path, img_name+"_lmk.txt")
        with open(saved_path, 'w') as fp:
            lmk_len = len(lmk)
            for i, coord in enumerate(lmk):
                fp.write(str(coord[0])+" "+str(coord[1]))
                if not (lmk_len - 1 == i):
                    fp.write("\n")
        return saved_path 
    
    def save(self, idx_or_idx_list):
        if isinstance(idx_or_idx_list, list):
            i_list = idx_or_idx_list
        elif isinstance(idx_or_idx_list, int):
            i_list = [idx_or_idx_list]
        self._init_meta()
        print(i_list)
        for index in i_list:
            if index < len(self.image_collection) and index >= 0:
                img_object = self.image_collection[index]
                if img_object.is_landmark_detected():
                    print(img_object.is_landmark_detected())
                    self._add_meta(img_object)
        self._meta_save()

        print(self.meta)



    
    def save_all(self):
        for i in range(len(self.image_collection)):
            self.save(i)
            

    def changed_lmk_listner(self, image_index, item):
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

    create_status_bar = pyqtSignal(int)
    progress = pyqtSignal(int, str)
    remove_status_bar = pyqtSignal()
    def __init__(self, parent, program_data) -> None:
        super().__init__(parent)
        self.mutex = QMutex()
        self.function_list = []
        self.program_data = program_data
        self.pp = parent

    def run(self):
        while True:
            if len(self.function_list):
                self.mutex.lock()
                f = self.function_list.pop(0)
                self.mutex.unlock()
                f()

    def load_detector(self, path = None):
        self.create_status_bar.emit(1)
        def wrapper():
            try:
                self.progress.emit(0, "loading start")
                self.program_data.image_collection.load_predictor(path)
                self.progress.emit(0, "loading end")
            except Exception as e:
                lab = self.pp.get_status_show_message()
                print(e)
                print("path")
                # lab("this funtionality is not allowed. because meta file is not loaded", 2000)
                lab(str(e), 2000)
            finally:
                self.remove_status_bar.emit()

        self.mutex.lock()
        self.function_list.append(wrapper)
        self.mutex.unlock()

    


    def save_data(self, id, index_list ):
        total_size = len(index_list)
        cancel_flag = False
        def cancel_f():
            nonlocal cancel_flag
            cancel_flag = True
        def wrapper():
            finished_index_list = []
            self.create_status_bar.emit(len(index_list))
            for i, index in enumerate(index_list):
                if not cancel_flag:
                    # self.program_data.save(index_list)
                    self.program_data.save(index)
                    finished_index_list.append(i)
                    self.progress.emit(i + 1, self.program_data[index].img_name + " finished.")
                else:
                    self.remove_status_bar.emit()
                    self.job_finished.emit(id,finished_index_list, False)
                    return
            self.remove_status_bar.emit()
            self.job_finished.emit(id,finished_index_list, True)

        self.mutex.lock()
        self.function_list.append(wrapper)
        self.mutex.unlock()
        return cancel_f

    def do_detect_lmk(self, id, index_list):
        total_size = len(index_list)
        cancel_flag = False
        def cancel_f():
            nonlocal cancel_flag
            cancel_flag = True
        def wrapper():
            finished_index_list = []
            self.create_status_bar.emit(len(index_list))

            for i, index in enumerate(index_list):
                if not cancel_flag:
                    self.program_data[index].redetect_flag = True
                    self.program_data[index].calc_lankmark_from_dlib(*(self.program_data.get_detector_and_predictor()))
                    finished_index_list.append(i)
                    self.lmk_load_finished.emit(index)
                    self.progress.emit(i + 1, self.program_data[index].img_name + " finished.")
                else:
                    self.remove_status_bar.emit()
                    self.job_finished.emit(id,finished_index_list, False)
                    return
            self.remove_status_bar.emit()
            self.job_finished.emit(id,finished_index_list, True)
        self.mutex.lock()
        self.function_list.append(wrapper)
        self.mutex.unlock()

        return cancel_f

    def do_load_image(self, id, index_list):
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
                    self.image_load_finished.emit(index)
                else:
                    self.job_finished.emit(id, finished_index_list, False)
            self.job_finished.emit(id, finished_index_list, True)
        
        self.mutex.lock()
        self.function_list.append(wrapper)
        self.mutex.unlock()
        return cancel_f





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
        # self.test.load("./test.JPG")
        # self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        # self._scene.addPixmap()
        self.img_overlay = self._scene.addPixmap(self.test)
        
        
        # draw color
        self.pen = QtGui.QPen()
        self.brush = QtGui.QBrush(QtGui.QColor(0,0,0))
        self.point = QtGui.QBrush(QtGui.QColor(0,0,0))

        self.outer_upper_side_brush = QtGui.QPen(QtGui.QColor(255,0,0))
        self.inner_upper_side_brush = QtGui.QPen(QtGui.QColor(0,255,255))
        self.inner_lower_side_brush = QtGui.QPen(QtGui.QColor(255,255,0))
        self.outer_lower_side_brush = QtGui.QPen(QtGui.QColor(0,0,255))
        
        


        # self.reset_image_configuration()
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
        test = test.fromImage(qimage)
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
        

        for i, (ll) in enumerate(self.lines):
            p1, p2 = ll.connected_pts
            ll.setLine(p1.pos().x(), p1.pos().y(), p2.pos().x(), p2.pos().y())
            ll.setVisible(True)

    def reset_image_configuration(self):
        self.image_scale_factor = 1.0
        pixel = 10
        self.circle_list  = []
        self.lines = []
        import random
        
        # self.program_data.
        full_index = self.program_data.image_collection.full_index
        eye = self.program_data.image_collection.eye
        contour = self.program_data.image_collection.contour
        mouse = self.program_data.image_collection.mouse
        eyebrow = self.program_data.image_collection.eyebrow
        nose = self.program_data.image_collection.nose
        for i in range(len(full_index)):
            test1 = random.randrange(0, 200)
            test2 = random.randint(0, 200)
            circle = self._scene.addEllipse(QtCore.QRectF(-pixel/2, -pixel/2, pixel, pixel),self.pen, self.brush)
            circle.setPos(test1, test2)
            circle.setZValue(1)
            circle.num = i
            circle.connected_line = set()
            self.circle_list.append(circle)
        
        def create_line(index_list, pen):
            for i, (p_i, p_j) in enumerate(zip( index_list[:-1], index_list[1:] )):
                t1 = self.circle_list[p_i]
                t2 = self.circle_list[p_j]
                line = QtCore.QLineF(t1.x(), t1.y(), t2.x(), t2.y())
                ll = self._scene.addLine(line, pen)
                t1.connected_line.add(ll)
                t2.connected_line.add(ll)
                ll.connected_pts = []
                ll.connected_pts.append(t1)
                ll.connected_pts.append(t2)
                self.lines.append(ll)     

        contour_full = contour['full_index']
        l_eye_upper = eye['left_eye']['upper_eye']
        l_eye_lower = eye['left_eye']['lower_eye']
        r_eye_upper = eye['right_eye']['upper_eye']
        r_eye_lower = eye['right_eye']['lower_eye']
        outer_upper_mouse = mouse['outer_upper_mouse']
        outer_lower_mouse = mouse['outer_lower_mouse']
        inner_upper_mouse = mouse['inner_upper_mouse']
        inner_lower_mouse = mouse['inner_lower_mouse']

        nose_vertical = nose['vertical']
        nose_hoizontal = nose['horizontal']

        left_eyebrow = eyebrow['left_eyebrow']['full_index']
        right_eyebrow = eyebrow['right_eyebrow']['full_index']

        create_line(left_eyebrow, self.outer_lower_side_brush)
        create_line(right_eyebrow, self.outer_lower_side_brush)

        create_line(nose_vertical, self.outer_upper_side_brush)
        create_line(nose_hoizontal, self.outer_lower_side_brush)

        create_line(contour_full, self.outer_upper_side_brush)
        create_line(l_eye_upper, self.outer_upper_side_brush)
        create_line(l_eye_lower, self.outer_lower_side_brush)
        create_line(r_eye_upper, self.outer_upper_side_brush)
        create_line(r_eye_lower, self.outer_lower_side_brush)
        create_line(outer_upper_mouse, self.outer_upper_side_brush)
        create_line(outer_lower_mouse, self.outer_lower_side_brush)
        create_line(inner_upper_mouse, self.inner_upper_side_brush)
        create_line(inner_lower_mouse, self.inner_lower_side_brush)
        # for i, (t1,t2) in enumerate(zip(self.circle_list[:-1], self.circle_list[1:])):
        #     e = t1.x()
        #     line = QtCore.QLineF(t1.x(), t1.y(), t2.x(), t2.y())
        #     ll = self._scene.addLine(line, self.pen)
        #     t1.connected_line.add(ll)
        #     t2.connected_line.add(ll)
        #     ll.connected_pts = []
        #     ll.connected_pts.append(t1)
        #     ll.connected_pts.append(t2)
        #     self.lines.append(ll)     

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
            if self.itemAt(vp) == self.img_overlay:
                sp = self.mapToScene(vp)
                lp = self.img_overlay.mapFromScene(sp).toPoint()
                # self.pixmapClicked.emit(lp)
            elif self.itemAt(vp) in self.circle_list :
                self.selected_pts = self.itemAt(vp)
            else : # this case line 
                sp = self.mapToScene(vp)
                lp = self.img_overlay.mapFromScene(sp).toPoint()
                
        elif event.button() == QtCore.Qt.RightButton:
                self.loc = vp
                self.right_mouse_pressed = True

        super(ImageWidget, self).mousePressEvent(event)
        
    def mouseReleaseEvent(self, event) -> None:
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
        return super().mouseReleaseEvent(event)
    
    def mouseMoveEvent(self, e):  # e ; QMouseEvent
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
            if self.selected_pts :
                sp = self.mapToScene(e.pos())
                lp = self.img_overlay.mapFromScene(sp).toPoint()
                self.selected_pts.setPos(lp.x(), lp.y())
                self.circle_line_edit(self.selected_pts)
                self.lmk_data_changed_signal.emit(self.index, self.selected_pts)
            
    def wheelEvent(self, e):  # e ; QWheelEvent
        ratio = e.angleDelta().y() / self.angle_ratio
        scale = 1.0
        scale +=  self.scale_increase_size * (ratio)
        self.scale(scale, scale)

class InspectorSignalCollection(QObject):
    InspectorIndexSignal = pyqtSignal(int)
    InspectorLmkDetectSignal = pyqtSignal(int)
    InspectorSaveSignal = pyqtSignal(int)
    InspectorLoadMetaData = pyqtSignal()
    
        

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


        # def update_when_finished_cur_image(i):
        #     print("tal")
        #     print(i)
        #     print(self.program_data.get_cur_index())
        #     if i == self.program_data.get_cur_index():
        #         print("test")
        #         self.signal.InspectorIndexSignal.emit(self.program_data.get_cur_index())
        # self.worker.image_load_finished.connect(update_when_finished_cur_image)

        def check_and_load_meta_file(name):
            self.input_data['root_directory'][0].setText(name)
            if osp.exists(osp.join(name, "meta.yaml")):
                self.program_data.load(name, "ict_lmk_info.yaml")
                self.data['meta file'].setText("meta.yaml")
                j_id = int(time.time())
                self.jobs[j_id] = False
                cancel_f = self.worker.do_load_image(j_id, [self.program_data.get_cur_index()])
                self.current_job_item = (j_id , cancel_f)
                self.upate_ui()
                self.signal.InspectorLoadMetaData.emit()
                self.worker.load_detector()
            else:
                self.data["meta file"].setText("Not Found.")

     
        self.input_data['root_directory'][1].clicked.connect(self.open_and_find_directory(check_and_load_meta_file))

        def setup_save_loc():
            self.open_and_find_directory(lambda name : self.input_data['save root location'][0].setText(name))()
            self.program_data.save_location = self.input_data['save root location'][0].text()


        self.input_data['save root location'][1].clicked.connect(setup_save_loc)
        
        

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
    

        def update_when_finished_cur_image(i):
                if i == self.program_data.get_cur_index():
                    self.signal.InspectorIndexSignal.emit(self.program_data.get_cur_index())
        self.worker.image_load_finished.connect(update_when_finished_cur_image)
        def prev():
            lab = self.pp.get_status_show_message()

            try:
                j_id = int(time.time())
                self.program_data.dec_index()
                self.jobs[j_id] = False
                cancel_f = self.worker.do_load_image(j_id, [self.program_data.get_cur_index()])
                self.current_job_item = (j_id , cancel_f)
                self.upate_ui()
            except:
                lab("this funtionality is not allowed. because meta file is not loaded", 2000)


            
        def next():
            lab = self.pp.get_status_show_message()

            try:
                j_id = int(time.time())
                self.program_data.inc_index()
                self.jobs[j_id] = False
                cancel_f = self.worker.do_load_image(j_id, [self.program_data.get_cur_index()])
                self.current_job_item = (j_id , cancel_f)
                self.upate_ui()

            except:
                lab("this funtionality is not allowed. because meta file is not loaded", 2000)


        self.prev.clicked.connect( prev )
        self.next.clicked.connect( next )

        def update_when_finished_cur_lmk(i):
                idx = self.program_data.get_cur_index()
                if i == idx:
                    self.signal.InspectorLmkDetectSignal.emit(idx)

        self.worker.lmk_load_finished.connect(update_when_finished_cur_lmk)
        def detect(index_list = None):
            idx = self.program_data.get_cur_index()
        
            lab = self.pp.get_status_show_message()
            try:
                if index_list == None:
                    index_list = [idx]
                

                if self.current_job_item == None :
                    j_id = int(time.time())
                    self.jobs[j_id] =False
                    cancel_f = self.worker.do_detect_lmk(j_id, index_list)
                    self.current_job_item = (j_id , cancel_f)
                else:
                    self.current_job_item[1]() # cancel function
                    j_id = int(time.time())
                    self.jobs[j_id] = False
                    cancel_f = self.worker.do_detect_lmk(j_id, index_list)
                    self.current_job_item = (j_id , cancel_f)
            except Exception as e :
                lab("this funtionality is not allowed. because meta file is not loaded", 2000)
                lab(str(e), 2000)

        def detect_current():
            detect()

        def detect_all():
            lab = self.pp.get_status_show_message()

            try:
                detect(list(range(len(self.program_data))))
                # detect(list(range(3)))
            except:
                lab("detect_all() : this funtionality is not allowed. because meta file is not loaded.", 2000)      
                
        def detect_finished_function(j_id, index_list, flag):
            # emitted by thread
            if flag == False : #when canceled
                delete_f = self.jobs.pop(j_id)
            if flag == True :
                self.current_job_item = None 
                delete_f = self.jobs.pop(j_id)
        self.worker.job_finished.connect(detect_finished_function)

        self.detect_all_button.clicked.connect(detect_all)
        self.detect_button.clicked.connect(detect_current)


        def save_function():
            lab = self.pp.get_status_show_message()
            try:
                j_id = int(time.time())
                self.jobs[j_id] =False                
                self.worker.save_data(j_id, [self.program_data.get_cur_index()])
            except:
                lab("this funtionality is not allowed. because meta file is not loaded.", 2000)      

        def save_all_function():
            lab = self.pp.get_status_show_message()
            try:
                j_id = int(time.time())
                self.jobs[j_id] =False                   
                print("len(self.program_data)", len(self.program_data))  
                self.worker.save_data(j_id, list(range(len(self.program_data))))
                    # self.signal.call_save_signal(i)
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

        self.connect_GUI_to_thread()

    def connect_GUI_to_thread(self):
        self.worker_thread.create_status_bar.connect(self.make_status_progress)
        self.worker_thread.remove_status_bar.connect(self.remove_status_progress)
        self.worker_thread.progress.connect(self.write_status_progress)

    
    def write_status_progress(self, value, msg):
        self.progress_bar.setValue(value)
        self.status_label.setText(msg)
        

    def make_status_progress(self, end):
        self.progress_bar.setRange(0, end)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

    def remove_status_progress(self):
        self.status_label.setText("")
        self.progress_bar.setVisible(False)

    def get_status_show_message(self):
        return self.statusbar.showMessage

    def connect_widgets_functionality(self):
        def wrapper(i):
            self.imagewidget.set_image_index(i)
            self.imagewidget.reload_image()
            self.imagewidget.reload_lmk_to_view()
        def load_meta():
            self.imagewidget.reset_image_configuration()
        
        self.inspectorwidget.signal.InspectorLoadMetaData.connect(load_meta)
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