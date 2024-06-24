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

class ImageViewWidget(QGraphicsView):
    def __init__(self, program_data: datamanager.DataManager, worker : qtthread.Worker):

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


class ImageViewWidget(QGraphicsView):
    lmk_data_changed_signal = pyqtSignal(int, QGraphicsEllipseItem)
    def __init__(self, parent, program_data: datamanager.DataManager):

        self._scene = QGraphicsScene()
        super().__init__(self._scene)

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

if __name__ == "__main__":


    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    form = ImageViewWidget()
    form.show()
    exit(app.exec_())