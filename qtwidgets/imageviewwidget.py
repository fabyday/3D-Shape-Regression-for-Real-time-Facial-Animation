import typing
from PyQt5.QtWidgets import *
import time
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QImage
from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal, QThread, QMutex
import sys
import ict_fact_meta
import ctypes
import os.path as osp
import os 
import cv2
from PyQt5.QtCore import Qt

import yaml
import qtthread
import metadata 
import datamanager
import uuid
import logger 
image_logger = logger.root_logger.getChild("image view")
from enum import Enum
class ImageEditMode(Enum):
    SELECT = 1
    TRANSLATION = 2 
    ROTATION = 3


class SELECT_TYPE(Enum):
    SELECT_DEFAULT = 1
    SELECT_MULTIPLE = 2
    SELECT_CONNECTED = 3



class ImageViewWidget(QGraphicsView):
    lmk_data_changed_signal = pyqtSignal(int, QGraphicsEllipseItem)
    def __init__(self, parent, ctx: datamanager.DataManager ):

        self._scene = QGraphicsScene()
        super().__init__(self._scene)

        self.m_ctx = ctx 
        self.m_mode = ImageEditMode.SELECT
        self.m_select_mode = SELECT_TYPE.SELECT_DEFAULT
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
        
        self.m_color_list = [self.inner_upper_side_brush, 
                             self.inner_lower_side_brush,
                             self.outer_upper_side_brush,
                             self.outer_lower_side_brush,]


        # self.reset_image_configuration()
        self.angle_ratio = 120
        self.scale_increase_size = 0.2
        
        self.selected_pts = None 
        self.right_mouse_pressed = False
        self.left_mouse_pressed = False


        self.index = -1
        image_logger.info("image view was initialized")


        parent.selected_data_changed_signal.connect(self.reload_image)

    #qt callback function
    @pyqtSlot(uuid.UUID)
    def reload_image(self, image_uuid):
        image = self.m_ctx[image_uuid].m_image
        if image.image is None :
            return 
        if len(image.image)<3:
            frame = cv2.cvtColor(image.image, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(image.image, cv2.COLOR_BGR2RGB)
        
        (w,h) = image.shape
        bytesPerLine = 3 * w

        qimage = QImage(frame.data, w, h, bytesPerLine, QImage.Format.Format_RGB888)
        test = QtGui.QPixmap()
        test = test.fromImage(qimage)
        # self.img_overlay.pixmap().convertFromImage(qimage)
        self.img_overlay.setPixmap(test)
        image_logger.debug("action : reload_image ")
        self.reload_lmk_to_view(image_uuid)
    
    @pyqtSlot()
    def create_landmark(self):
        lmk_structure_meta = self.m_ctx.get_landmark_structure_meta()
        


    # this method will be called by callback and image load function
    def reload_lmk_to_view(self, image_uuid):
        lmks = self.m_ctx[image_uuid].m_lmk.landmark
        
        if lmks is None :
            self.circle_line_visible(False)
            return
        else :
            self.circle_line_visible(True)


        for lmk, circle in zip(lmks, self.circle_list):
            x = lmk[0]
            y = lmk[1]
            circle.setPos(x, y)
            circle.setVisible(True)
        

        for i, (ll) in enumerate(self.lines):
            p1, p2 = ll.connected_pts
            ll.setLine(p1.pos().x(), p1.pos().y(), p2.pos().x(), p2.pos().y())
            ll.setVisible(True)
        image_logger.debug("action : reload landmark to view ")

    def reset_image_configuration(self):
        image_logger.debug("action : reset image configuration ")
        self.image_scale_factor = 1.0
        pixel = 10
        self.circle_list  = []
        self.lines = []
        import random


        def create_line(color_index, vert_indice):
            index_list = vert_indice
            pen = self.m_color_list[color_index]

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
        def rec_(lmk_struct , key_list):
            for key in key_list:
                component = lmk_struct[key]
                if not component.is_end_component():
                    rec_(component, component.keys())
                else :
                    create_line(component.get_type().value, component.get_indice_list())
    

        landmark_structure = self.m_ctx.get_landmark_structure_meta()
        full_index = landmark_structure.get_full_index()
        for i in range(len(full_index)):
            test1 = random.randrange(0, 200)
            test2 = random.randint(0, 200)
            circle = self._scene.addEllipse(QtCore.QRectF(-pixel/2, -pixel/2, pixel, pixel),self.pen, self.brush)
            circle.setPos(test1, test2)
            circle.setZValue(1)
            circle.num = i
            circle.connected_line = set()
            self.circle_list.append(circle)
        

        landmark_structure = self.m_ctx.get_landmark_structure_meta()
        rec_(landmark_structure, landmark_structure.component_name_list())
    
        self.circle_line_visible(False)


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
        self.test = QtGui.QPixmap()
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

        super(ImageViewWidget, self).mousePressEvent(event)
        
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


    def keyPressEvent(self, e):
        key = e.key()
        template_logging_str = "image viewer action : %s"
        if key == ord("R"):
            image_logger.info(template_logging_str,  "rotation mode")
            self.m_mode = ImageEditMode.ROTATION
        elif key == ord("G"):
            image_logger.info (template_logging_str , "translation mode")
            self.m_mode = ImageEditMode.TRANSLATION
        elif key == ord("S") : 
            image_logger.info(template_logging_str, "select mode")
            self.m_mode = ImageEditMode.SELECT
        
        if self.m_mode == ImageEditMode.SELECT:
            modifiers = e.modifiers()

            if modifiers == QtCore.Qt.ShiftModifier:
            
                self.m_select_mode = SELECT_TYPE.SELECT_MULTIPLE
                image_logger.info(template_logging_str, "multiple pts select mode")
            elif modifiers == QtCore.Qt.AltModifier:

                self.m_select_mode = SELECT_TYPE.SELECT_CONNECTED
                image_logger.info(template_logging_str, "connected pts select mode")
            elif modifiers == QtCore.Qt.ControlModifier:
                pass 
            elif modifiers == QtCore.Qt.MetaModifier:
                pass 
    def keyReleaseEvent(self, e):
        key = e.key()
        template_logging_str = "image viewer action : %s"
        if self.m_mode == ImageEditMode.SELECT:
            if key == QtCore.Qt.ShiftModifier:
                self.m_select_mode = SELECT_TYPE.SELECT_DEFAULT
            elif key == QtCore.Qt.AltModifier:
                self.m_select_mode = SELECT_TYPE.SELECT_DEFAULT
            image_logger.info(template_logging_str, "reset default select mode")
if __name__ == "__main__":


    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    form = ImageViewWidget()
    form.show()
    exit(app.exec_())