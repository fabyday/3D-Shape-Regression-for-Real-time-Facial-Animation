from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import sys
from collections import deque

class GraphicsScene(QGraphicsScene):
    def __init__(self, parent=None, ns=""):
        super(GraphicsScene, self).__init__(parent)
        self.parent = parent
        self._selectedItemVec = deque()
        self.ns = ns
        self.setBackgroundBrush(QBrush(QColor(50,50,50) , Qt.SolidPattern))

    def mouseReleaseEvent(self, event):
        item = self.itemAt(event.scenePos ().x(), event.scenePos ().y())
        if item:
            item.setSelected(1)
        else:
            if len(self._selectedItemVec):
                self._selectedItemVec.popleft()
                return QGraphicsScene.mouseReleaseEvent(self, event)
        if event.modifiers() & Qt.ShiftModifier:
            for item in self._selectedItemVec:
                item.setSelected(1)
        else:
            self._selectedItemVec.popleft()

    def mousePressEvent(self, event):

        item = self.itemAt(event.scenePos ().x(), event.scenePos ().y())
        if item:
            item.setSelected(1)
            self._selectedItemVec.append(item)
        else:
            return QGraphicsScene.mousePressEvent(self, event)


class MainClass(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)        

        self.setGeometry(50,50,400,600)

        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QGridLayout()
        # layout.setMargin(3)
        layout.setContentsMargins(3,3,3,3)

        widget.setLayout(layout)

        view = QGraphicsView()
        view.setMouseTracking(1)
        view.setDragMode(QGraphicsView.RubberBandDrag)
        view.setRenderHint(QPainter.Antialiasing)
        view.setRenderHint(QPainter.TextAntialiasing)
        view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse) 
        view.setAlignment(Qt.AlignJustify)


        scene = GraphicsScene(self)
        scene.setSceneRect(0,0, 300, 500)
if __name__ == "__main__":


    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    form = MainClass()
    form.show()
    exit(app.exec_())