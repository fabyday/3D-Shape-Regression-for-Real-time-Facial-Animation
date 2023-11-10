from pyqt import QtCore      # core Qt functionality
from pyqt import QtGui       # extends QtCore with GUI functionality
from pyqt import QtOpenGL    # provides QGLWidget, a special OpenGL QWidget

import OpenGL.GL as gl        # python wrapping of OpenGL
from OpenGL import GLU        # OpenGL Utility Library, extends OpenGL functionality
import OpenGL.arrays.vbo as glvbo
import sys                    # we'll need this later to run our Qt application
# PyOpenGL imports



class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)    # call the init for the parent class
        self.resize(800, 800)
        self.setWindowTitle('Optimizer Visualizer')


class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        self.parent = parent
        QtOpenGL.QGLWidget.__init__(self, parent)

    def initializeGL(self):
        self.qglClearColor(QtGui.QColor(0, 0, 255))    # initialize the screen to blue
        gl.glEnable(gl.GL_DEPTH_TEST)                  # enable depth testing
    def resizeGL(self, width, height):
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = width / float(height)

        GLU.gluPerspective(45.0, aspect, 1.0, 100.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)




    def set_proj_cam(self, Q, Rt):
        pass

    def set_image(self, img):
        pass

    def write_mesh_to_image(self, img, v, f):
        pass 



    def write_contour_to_image(self, img, v):
        pass 



    def write_vertex_to_image(self, img, pts2d):
        pass


    def write_boundary_to_image(self, img, pts2d):
        pass





def run():
    app = QtGui.QApplication(sys.argv)

    win = MainWindow()
    win.show()

    sys.exit(app.exec_())



run()