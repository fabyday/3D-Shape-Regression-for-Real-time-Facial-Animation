import logging.handlers
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
import time
import yaml
import queue
import uuid
import logging
import signals


thread_logger = logging.getLogger("workerLogger")
thread_logger.setLevel(logging.DEBUG)
# RotatingFileHandler
log_max_size = 10 * 1024 * 1024  ## 10MB
log_file_count = 20
log_path = "./logs"
if not os.path.exists(log_path):
    os.makedirs(log_path)


rotatingFileHandler = logging.handlers.RotatingFileHandler(
    filename= os.path.join(log_path, 'output.log'),
    maxBytes=log_max_size,
    backupCount=log_file_count
)
thread_logger.addHandler(rotatingFileHandler)
class Runnable():
    def __init__(self):
        self.m_id = uuid.uuid4()
        self.m_cancel = False 
        self.m_is_completed = False
        self.m_callback = None

    def cancel(self):
        self.m_cancel = True


    def then(self, func):
        """
            return this runnable object
        """
        self.m_callback = func

    def is_completed(self):
        return self.m_is_completed

    def _run(self):
        if not self.m_cancel:
            thread_logger.debug("uuid : {} function _run".format(self.m_id))
            self.run()
            self.m_is_completed = True
        return self.is_completed()

    @property 
    def uid(self):
        return self.m_id
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.m_is_completed:
            raise StopIteration()
        else:
            return self
    
    def __len__(self):
        return 1

    # user implementation
    def run(self):
        print("not impl")
        return NotImplemented
    

class Job(Runnable):
    def __init__(self, func):
        super().__init__()
        self.m_func = func
    def run(self):
        return self.m_func()
    
class Jobs(Runnable):
    def __init__(self):
        super().__init__()
        self.m_runnable_list = []



    def add(self, runnable:Runnable):
        self.m_runnable_list.append(runnable)

    def run(self):
        thread_logger.debug("uuid : {} class : jobs function : run() ".format(self.m_id))
        for runnable in self.m_runnable_list:
            runnable._run()

    def __len__(self):
        return len(self.m_runnable_list)

    def __iter__(self):
        return iter(self.m_runnable_list)

    def cancel(self):
        for runnable in self.m_runnable_list:
            if not runnable.is_completed():
                runnable.cancel()



class Worker(QThread):

    class NotRunnableObjectException(Exception):
        pass 

    jobs_complete_signal = pyqtSignal(uuid.UUID, signals.Event)
    job_complete_signal = pyqtSignal(uuid.UUID) #return uuid
    job_cancel_signal = pyqtSignal(uuid.UUID)  #return uuid

    create_status_bar_signal =pyqtSignal(int, int) # init value, length 
    job_progress_signal = pyqtSignal(uuid.UUID, int, int, bool, str) # curjob uuid, cur_loc, length, completed or failed, msg
    remove_status_bar_signal =pyqtSignal() # init value, length 

    # create_status_bar_signal => job_progress_signal  => remove_status_bar signal

    def __init__(self, parent):
        """
            parent : UI Object
        """
        super().__init__(parent)
        self.jobs_queue = queue.Queue()
        self.m_parent = parent
        self.m_mutex = QMutex()

    def reserve_job(self, job_o_joblist : Runnable, job_finish_event_type = signals.Event()):
        if isinstance(job_o_joblist, Runnable):
            job = job_o_joblist
        else:
            raise Worker.NotRunnableObjectException("one of job is not runnable")

        self.m_mutex.lock()
        self.jobs_queue.put((job, job_finish_event_type))
        self.m_mutex.unlock()
        thread_logger.debug("thread job was reserved ")


    def run(self):
        format_string = "{}-th item is {}"
        while True : 
            self.m_mutex.lock()
            thread_logger.debug("run thread : job queue size {} ".format(self.jobs_queue.qsize()))
            data = None 
            try :
                data = self.jobs_queue.get_nowait()
            except:
                pass 
            self.m_mutex.unlock()
            if data is None :
                time.sleep(0.0001)
                continue

            (data, event) = data 

            for cur_i, runnable in enumerate(data):
                thread_logger.debug("job task progress : {}/{}".format(cur_i + 1, len(data)))
                ret = runnable._run()
                one_base_index = cur_i + 1
                if ret : 
                    
                    self.job_complete_signal.emit(data.uid)
                    self.job_progress_signal.emit(data.uid, one_base_index, len(data), True, format_string.format(one_base_index, "completed."))
                else : 
                    self.job_cancel_signal.emit(data.uid)
                    self.job_progress_signal.emit(data.uid, one_base_index, len(data), True, format_string.format(one_base_index, "canceled."))
            if data.m_callback is not None :
                thread_logger.debug("job complete and run callback function.")
                data.m_callback(data)
            self.jobs_complete_signal.emit(data.m_id, event)

class DummyJob(Runnable):
    def __init__(self):
        super().__init__()



    
    def run(self):
        print(self.uid)
        print('test')




if __name__ == "__main__":
    class MyWindow(QMainWindow):
        def __init__(self):
            super().__init__()

            self.a = Worker(None)
            self.a.start()
            self.a.job_cancel_signal.connect(self.cons)
            self.a.job_complete_signal.connect(self.consa)
            self.a.job_progress_signal.connect(self.asd)
            self.btn = QPushButton(self)
            self.btn.clicked.connect(self.clicked)


        @pyqtSlot()
        def clicked(self):
            self.a.reserve_job(DummyJob())


        @pyqtSlot(int)
        def timeout(self, uid):
            print(uid)


        @pyqtSlot(uuid.UUID, int, int, bool, str)
        def asd(self,uuid, cur, length, comp, msg):
            print(uuid)
            print("{} / {} \n{} : {}".format(cur, length, comp, msg))


        @pyqtSlot(uuid.UUID)
        def cons(self,w):
            print("fails")
            print(w)
        @pyqtSlot(uuid.UUID)
        def consa(self, w):
            print("comope")
            print(w)
        
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    app.exec_()
        
    # while True:
    #     s = DummyJob()
    #     # s.cancel()
    #     a.reserve_job(s)

    # from PyQt5.QtWidgets import QApplication

    # app = QApplication(sys.argv)
    # form = ConfigurationEditor()
    # form.show()
    # exit(app.exec_())