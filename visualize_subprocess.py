import subprocess_helper as subp
# train data visualizer
# check afgter preprop coordinate descent copy


import geo_func as geo 
import os

import cv2, yaml 
import numpy as np 
import tqdm 
import sys
import visualizer as vis

from PyQt5.QtWidgets import *
import inspector_helper as ihelper
import typing
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QImage
from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal, QThread, QMutex, QTimer, Qt
import ctypes 



lmk_data_transfer = subp.LandmarkDataTransfer()


lmk_data_transfer.recv()



def load_data_train(path, start = None, end = None, lmk_indices = None, desired_img_width = None):
    meta_path = os.path.join(path, "meta.txt")
    with open(meta_path, 'r') as fp :
        meta = yaml.load(fp, yaml.FullLoader)
        image_ext = meta['image_extension']
        image_root = meta["image_root_location"]
        S_Rtinv_index_list_path = meta["S_Rtinv_index_list"]
        Q = np.load(os.path.join(path,meta['Q_location']))
        image_location = meta['image_name_location']
        Rt_inv_location = meta['Rt_inv_location']
        S_init_location = meta['S_init_location']
        S_location = meta['S_location']
        S_original_location = meta['S_original_location']
        root_path = meta['data_root']

    data_root = os.path.join(path, root_path)
    S_Rtinv_index_list = np.load(os.path.join(data_root,S_Rtinv_index_list_path))
    img_names = np.load( os.path.join(data_root, image_location))
    Ss = np.load(os.path.join(data_root, S_location))
    S_original = np.load(os.path.join(data_root, S_original_location))
    S_inits = np.load(os.path.join(data_root, S_init_location))
    Rt_invs = np.load(os.path.join(data_root, Rt_inv_location))
    res = [] 

    image_dict = {}
    img_names2 = list(set(img_names))
    resize_ratio = 1.0
    for name in img_names2:
        img = cv2.imread(os.path.join(image_root, name+image_ext ))
        if desired_img_width != None :
            h, w, _ = img.shape
            resize_ratio = desired_img_width/w
            img = cv2.resize(img, (int(resize_ratio*w),int(resize_ratio*h)))
        image_dict[name] = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img)

    if lmk_indices == None :
        lmk_indices = list(range(Ss.shape[1]))

    Ss = np.array(Ss)[: , lmk_indices, :]

    # for img_name, s, s_init, Rt_inv in tqdm.tqdm(zip(img_names, Ss, S_inits, Rt_invs), "load data"):
    for img_name,  s_init, S_Rt_index in tqdm.tqdm(zip(img_names, S_inits, S_Rtinv_index_list), "load data"):
        # res.append({'img' : image_dict[img_name][0], 'color_img' : image_dict[img_name][1], "S_index" : S_Rt_index, "S_init" : s_init[lmk_indices, :], "Rt_inv_index" : S_Rt_index})
        res.append({'img' : image_dict[img_name][0], 'color_img' : image_dict[img_name][1], "S_index" : S_Rt_index, "S_init" : s_init, "Rt_inv_index" : S_Rt_index})
    return Q, res[start:end], Ss, S_original, Rt_invs, resize_ratio
    



class ImageViewWidget(QGraphicsView):
    def __init__(self):
        self._scene = QGraphicsScene()
        super().__init__(self._scene)

        # bytesPerLine = 3 * w
        self.test = QtGui.QPixmap()
        self.img_overlay = self._scene.addPixmap(self.test)


        # qimage = QImage(frame.data, w, h, bytesPerLine, QImage.Format.Format_RGB888)
        # test = QtGui.QPixmap()
        # test = test.fromImage(qimage)
        # # self.img_overlay.pixmap().convertFromImage(qimage)
        # self.img_overlay.setPixmap(test)
        


    def edit(self, image_ndarray):

        image = image_ndarray
        h, w, *_ = image.shape
        if image_ndarray is None :
            return 
        if len(image.shape)<3:
            frame = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        bytesPerLine = 3 * w

        qimage = QImage(frame.data, w, h, bytesPerLine, QImage.Format.Format_RGB888)
        test = QtGui.QPixmap()
        test = test.fromImage(qimage)
        self.img_overlay.setPixmap(test)

    def wheelEvent(self, e):  # e ; QWheelEvent
        self.angle_ratio = 120
        self.scale_increase_size = 0.2
        ratio = e.angleDelta().y() / self.angle_ratio
        scale = 1.0
        scale +=  self.scale_increase_size * (ratio)
        self.scale(scale, scale)
 
def proj(Q, Rt, p):
    M = Rt [:3, :3]
    t = Rt [:, -1, np.newaxis]
    pts_2d_h = (Q @ (M @ p.T + t)).T
    pts_2d = pts_2d_h[:, :-1] / pts_2d_h[:, -1, np.newaxis]
    return pts_2d


import uuid 

# data num, fern, weak num slide
#
#
class Context(QObject):
    data_initialized = pyqtSignal(uuid.UUID)
    data_updated = pyqtSignal(uuid.UUID)
    

    def __init__(self):
        self.m_mutex = QMutex()        
        self.m_data = {}
        self.m_index_info_cursor = {}
        self.m_meta = {}


    class WeakNumberWiseIterator():
        def __init__(self):
            pass 

        def __iter__(self):
            return self 
        
        def __next__(self):
            pass 

    class FernNumberWiseIterator():
        def __init__(self):
            pass 

        def __iter__(self):
            pass 

        def __next__(self):
            pass 
    
    class ElementWiseIterator():
        def __init__(self):
            pass 

        def __iter__(self):
            pass 

        def __next__(self):
            pass 


    def get_iterator(self, iter_type):
        """
        "element"
        "fern"
        "weak"
        """
        if iter_type == "element":
            return Context.WeakNumberWiseIterator(self, self.m_index_info_cursor)
        elif iter_type == "weak":
            return Context.WeakNumberWiseIterator(self, self.m_index_info_cursor)
        elif iter_type == "fern":
            return Context.FernNumberWiseIterator(self, self.m_index_info_cursor)


    def get(self, uid : uuid.UUID):
        self.m_mutex.lock()
        data = self.m_data.get(uid, None)
        self.m_mutex.unlock()
        return data 
    

    def add_to_meta(self, data):
        if self.m_meta.get(data.header[lmk_data_transfer.Item.k_weak_num], None) is None :
            weak_data_collection = {}
            self.m_meta[data.header[lmk_data_transfer.Item.k_weak_num]] = weak_data_collection
            fern_data_collection = {}
            weak_data_collection[data.header[lmk_data_transfer.Item.k_fern_num]] = fern_data_collection
        else :
            weak_data_collection = self.m_meta.get(data.header[lmk_data_transfer.Item.k_weak_num])
            fern_data_collection = self.m_meta.get(data.header[lmk_data_transfer.Item.k_fern_num])

        fern_data_collection[lmk_data_transfer.Item.k_data_number] = {"uuid" : data.uid, "image_number" : data.header[lmk_data_transfer.Item.k_image_number]}
            


            
        

    def put(self, uid : uuid.UUID, item ):
        
        if self.m_data.get(uid, None ) is None :
            self.m_mutex.lock()
            self.m_data[uid] = item 
            self.m_mutex.unlock()

            self.add_to_meta(data=item)
        else :
            raise KeyError("you put the same key")



class RecvWorker(QThread):
    lmk_data_recived = pyqtSignal(uuid.UUID)
    info_data_recived = pyqtSignal(uuid.UUID)
    def __init__(self, ctx : Context, lmk_data_transfer : subp.LandmarkDataTransfer):
        super().__init__()
        self.m_lmk_data_transfer = lmk_data_transfer
        self.m_ctx = ctx

    def run(self):
        
        while True : 
            data = self.m_lmk_data_transfer.recv()
            if data is None :
                continue
            self.m_ctx.put(data.uid, data)
            self.lmk_data_recived.emit(data.uid)

            

             
            
            
            
            




    


class MyApp(QMainWindow):

    image_changed_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        # self.Q, self.data, self.Ss, self.S_original, self.Rt_invs, self.resize_ratio

        self.m_ctx = Context()
        
        self.m_thread = RecvWorker(self.m_ctx, lmk_data_transfer)
        self.m_thread.start()

        self.m_index = 0
        self.jump_size = 1
        self._init_ui()




    def _init_ui(self):
        width = ctypes.windll.user32.GetSystemMetrics(0)
        height = ctypes.windll.user32.GetSystemMetrics(1)
        self.setWindowTitle('data_preview')
        self.resize(int(width*0.8), int(height*0.8))
        self.move(self.screen().geometry().center() - self.frameGeometry().center())

        # status bar
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.status_label = QLabel()
        self.statusbar.addWidget(self.status_label)

        self.image_view = ImageViewWidget()
        self.image_changed_signal.connect(self.image_view.edit)
        # self.image_view.edit(self.get_image_from_index(self.m_index))

        self.make_slider_widget(self.data_slider_callback, {}, {}, {}, {})
        self.make_slider_widget(self.fern_slider_callback, {}, {}, {}, {})
        self.make_slider_widget(self.weak_slider_callback, {}, {}, {}, {})
        


        slider_layout1 = QHBoxLayout()
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.index_edit)
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)
        btn_layout.addStretch(1)
        
        self.prev_btn = QPushButton("prev")
        self.next_btn = QPushButton("next")
        self.index_edit = QLineEdit(str(self.m_index))
        
        self.jump_up_btn = QPushButton("jump scale up")
        self.jump_down_btn = QPushButton("jump scale down")
        self.jump_scale_edit = QLineEdit(str(self.jump_size))
        
        self.jump_up_btn.clicked.connect(lambda : self.jump_size_f(self.jump_size + 1, lambda : self.jump_scale_edit.setText(str(self.jump_size))))
        self.jump_down_btn.clicked.connect(lambda : self.jump_size_f(self.jump_size - 1,  lambda : self.jump_scale_edit.setText(str(self.jump_size))))
        self.jump_scale_edit.returnPressed.connect(self.edit_changed)        
        jump_btn_layout = QHBoxLayout()
        jump_btn_layout.addStretch(1)
        jump_btn_layout.addWidget(self.jump_scale_edit)
        jump_btn_layout.addWidget(self.jump_down_btn)
        jump_btn_layout.addWidget(self.jump_up_btn)
        jump_btn_layout.addStretch(1)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.index_edit)
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)
        btn_layout.addStretch(1)
        
        widget = QWidget()

    #     imagewidget.initUI()
    #     inspectorwidget.initUI()
        root_layout = QVBoxLayout(widget)
        self.setCentralWidget(widget)
        root_layout.addWidget(self.image_view)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.index_edit)
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)
        btn_layout.addStretch(1)



        root_layout.addLayout(btn_layout)
        root_layout.addLayout(jump_btn_layout)
        self.setLayout(root_layout)


        self.prev_btn.clicked.connect(self.prev)
        self.next_btn.clicked.connect(self.next)
        self.index_edit.returnPressed.connect(self.edit_changed)

    def make_slider_widget(self, callback_slider, callback_next, callback_prev, callback_edit_line):
        layout = QHBoxLayout()

        data_slider = QSlider()
        data_slider.setRange(0, 0)
        data_slider.setTickInterval(1)
        data_slider.valueChanged.connect(callback_slider)

        data_slider_next_btn =QPushButton("next")
        data_slider_next_btn.clicked.connect(callback_next)
        data_slider_prev_btn =QPushButton("prev")
        data_slider_prev_btn.clicked.connect(callback_prev)


        data_slider_edit = QLineEdit(str(0))
        data_slider_edit.returnPressed.connect(callback_edit_line)

        layout.addStretch(1)
        layout.addWidget(self.data_slider_edit)
        layout.addWidget(self.data_slider_prev_btn)
        layout.addWidget(self.data_slider_next_btn)
        layout.addWidget(self.data_slider)
        layout.addStretch(1)

        return layout, data_slider, data_slider_next_btn, data_slider_prev_btn, data_slider_edit

    def jump_size_f(self, size, callback = None ):
        new_jump_scale = size
        if new_jump_scale >= 1 or new_jump_scale < len(self.data)//2:
            self.jump_size = new_jump_scale
        else : 
            self.jump_size = 1
        

        if callback :
            callback()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()
        elif e.key() == Qt.Key_A:
            self.prev()
        elif e.key() == Qt.Key_D:
            self.next()
        elif e.key() == Qt.Key_Q:
            self.jump_size_f(self.jump_size + 1)
            self.jump_scale_edit.setText(str(self.jump_size))
        elif e.key() == Qt.Key_E:
            self.jump_size_f(self.jump_size-1)
            self.jump_scale_edit.setText(str(self.jump_size))
    

    def data_slider_callback():
        pass 
    def fern_slider_callback():
        pass 

    def weak_slider_callback():
        pass


    def get_valid_index(self, index):
        if index >= len(self.data):
            return 0
        if index < 0 :
            return len(self.data) - 1
        return index


    def get_image_from_index(self, i):
        img = self.data[i]["color_img"]
        h, *_ = img.shape

        S = self.data[i]["S_index"]
        S_pts = self.Ss[S]

        Rt_inv = self.data[i]["Rt_inv_index"]
        Rt_inv_mat = self.Rt_invs[Rt_inv]
        S_init = self.data[i]["S_init"]
        S_init = self.Ss[S_init]

        S_prj = proj(self.Q, Rt_inv_mat, S_pts)
        S_prj= geo.convert_to_cv_image_coord(S_prj, h)
        S_i_proj = proj(self.Q, np.eye(3,4, dtype=np.float32), S_init)
        S_i_proj = geo.convert_to_cv_image_coord(S_i_proj, h)

        new_img = vis.draw_circle(S_prj, img, colors=(0,0,255), radius=1)
        im = vis.draw_circle(S_i_proj, img, colors=(255,0,0), radius=2)
        im1 = vis.resize_img(im, 600)
        im2 = vis.resize_img(new_img, 600)
        im = vis.concatenate_img(1, 2, im2, im1)

        return im

    @pyqtSlot()
    def next(self):
        self.m_index = self.get_valid_index(self.m_index+self.jump_size)
        im = self.get_image_from_index(self.m_index)
       

        
        self.index_edit.setText(str(self.m_index))

        self.image_changed_signal.emit(im)         

    @pyqtSlot()
    def jump_size_edit_f(self):
        val = self.jump_scale_edit.text()
        try : 
            new_size = int(val)
        except:
            new_size = self.jump_size 
        finally:
            self.jump_size_f(new_size)
            self.jump_scale_edit.setText(str(self.jump_size))

    @pyqtSlot()
    def edit_changed(self):
        value = self.index_edit.text()
        try : 
            new_index = int(value)
        except:
            new_index = self.m_index
        finally:
            new_index = self.get_valid_index(new_index)
            self.index_edit.setText(str(new_index))
        
        
        self.image_changed_signal.emit(np.array([0]))
        



    @pyqtSlot()
    def prev(self):
        self.m_index = self.get_valid_index(self.m_index-self.jump_size)
        self.index_edit.setText(str(self.m_index))
        im = self.get_image_from_index(self.m_index)
        
        self.image_changed_signal.emit(im)
        



def run():
   app = QApplication(sys.argv)
   ex = MyApp()
   ex.show()
   sys.exit(app.exec_())

if __name__ == '__main__':
   run()
