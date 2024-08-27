import subprocess

import sys 
import os.path as osp 
from typing import *
 
import os
import sys 

import pickle
import threading 
import queue
import numpy as np 
import debugpy
import inspect 
import inspector_helper as insp
import time

import signal
import uuid 
class BaseDataTransfer: pass 

import logging 

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(name)s][%(threadName)s][%(levelname)s]: %(message)s')
handler.setFormatter(formatter)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(handler)


root_logger.debug("test")
class BaseDataTransfer:
    
    class BaseItem():
        MSG_TYPE = "base_type"
        def __init__(self, uid = None, header = None ):
            """
             Do not use constructor directly
            """
            self.m_msg_type = ""
            self.m_uid = uid
            if header is not None :
                # print(self.MSG_TYPE)
                print(header['message_type'])
                root_logger.info("%s %s", "message type", header['message_type'])
                if header['message_type'] != self.MSG_TYPE:
                    raise KeyError("message type is not fit.")
                else :
                    self.m_msg_type = header['message_type']
                    self.m_header = header

            else : 
                self.m_header = {"message_type" : self.MSG_TYPE} 


            print("clear")


        @property
        def header(self):
            return self.m_header 
        
        
        @classmethod
        def create(cls, buf):
            try : 
                if buf is None :
                    obj = cls(uuid.uuid4(), header=None )
                else:
                    header = pickle.load(buf)
                    print("test create")
                    obj = cls(uuid.uuid4(), header)
                    obj.deserialize(buf)
            except Exception as e :
                print(e)
                raise TypeError("this type is not fit.")
            return obj



        @property
        def uid(self):
            return self.m_uid

        def serialize(self):
            return NotImplemented
        
        def deserialize(self, buf):
            return NotImplemented


    def __init__(self, is_writer = False, python_file = '', msg_type = None ):
        self.m_pipe_connected_flag = False 
        self.m_python_file = python_file

        self.msg_type = msg_type

        self.is_writer = is_writer



        self.m_recv_callback = []

        self.m_queue = queue.Queue()
        
        self.m_cond = threading.Condition(threading.Lock())
        self.m_mutex = threading.Lock()
        self.m_wait_time_ratio = 1.0
        self.m_stdin = sys.stdin
        if self.is_writer:
            self.m_thread = threading.Thread(target = self.send_to_subprocess, name = "abs_threading")
            self.m_thread.daemon = True
            self.m_thread.start()
        else:
            self.m_thread = threading.Thread(target = self.recv_from_subprocess, name = "abs_threading")
            self.m_thread.daemon=True
            self.m_thread.start()

    def recv_from_subprocess(self):
        root_logger.info("recv thread is started now.")
        iter_num = 0
        iter_flag_num = 100
        while True : 
            data = self._recv()
            iter_num += 1 
            if iter_num % iter_flag_num == 0 : 
                root_logger.debug("%d %s", iter_num, "msg was received.")
            while True  : 
                wait_flag = True  
                self.m_mutex.acquire()
                try : 
                    self.m_queue.put_nowait(data)
                    wait_flag = False
                except queue.Full:
                    pass 
                    
                self.m_mutex.release()
                if wait_flag:
                    self.m_cond.acquire()
                    self.m_cond.wait(self.m_wait_time_ratio*0.5)
                    self.m_cond.release()
                else :
                    break

    def send_to_subprocess(self):
        root_logger.info("send thread is started now.")
        iter_num = 0 
        log_flag_num = 100
        while True : 
            
            wait_flag = False 
            self.m_mutex.acquire()
            try : 
                item = self.m_queue.get_nowait()
                self.m_wait_time_ratio = 1.0
            except queue.Empty:
                wait_flag = True 
            self.m_mutex.release()
            
            if wait_flag:
                self.m_cond.acquire()
                self.m_cond.wait(self.m_wait_time_ratio * 0.5)
                self.m_cond.release()
                self.m_wait_time_ratio *= 2 
                continue 
            self._send(item)
            iter_num+= 1
            if iter_num % 100 == 0:
                root_logger.debug("%d %s", iter_num, "time msg send.")



    @property
    def is_parent(self):
        return self.m_is_parent

    def create(self, *args):
        if self.is_writer:
            if osp.isabs(self.m_python_file):
                path = self.m_python_file
            else:
                path = osp.join(os.curdir, self.m_python_file)
            self.m_subp = subprocess.Popen(args=[ sys.executable, path], bufsize=0 ,stdin=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_CONSOLE)
            self.m_pipe_connected_flag = True
            
        else:
            pass
        self.add_system_call_handler()
        root_logger.info("program was successfully initialized.")

    def _send(self, data):
        try : 
            header, data = data.serialize()
            pickle.dump(header, self.m_subp.stdin)
            self.m_subp.stdin.write(data)
        except Exception as e:
            print(e)
            if self.m_subp.poll() is not None :
                pass 
        
    def send(self, data):
        flag = True 
        while flag :
            self.m_mutex.acquire()
            try : 
                self.m_queue.put_nowait(data)
                flag = False 
            except queue.Full :
                self.m_cond.notify()
            self.m_mutex.release()


    def _recv(self):
        
        new_message_object = None 
        try : 
            new_message_object = insp.MessageInstanceHelper.create_message_object_from_nested_class(self, BaseDataTransfer.BaseItem, self.m_stdin.buffer)
        except Exception as e :
            print(e)
        return new_message_object
        




    def recv(self):
        """
            if queue is empty return None 
        """
        self.m_mutex.acquire()
        try : 
            res= self.m_queue.get_nowait()
        except queue.Empty:
            res =  None
        self.m_mutex.release()
        return res
        

    def recv_callback(self, callback : Callable[[Any], None])-> None:
        self.m_recv_callback.append(callback)
    
    def close(self):
        print("kill!")
        self.m_subp.kill()
        if self.m_pipe_connected_flag:
            self.m_pipe_connected_flag = False 


    def add_system_call_handler(self, msg_enum  = None , handler = None):
        
        def ternmiate_handler(signum, frame):
            root_logger.info("terminate program.")
            if self.m_subp is not None : 
                self.m_subp.terminate()
                self.m_subp.wait()
            exit(0)


        if msg_enum is not None and handler is not None:
            signal.signal(msg_enum, handler)
        else : 
            signal.signal(signal.SIGBREAK, ternmiate_handler)
            signal.signal(signal.SIGABRT, ternmiate_handler)
            signal.signal(signal.SIGINT, ternmiate_handler)
            signal.signal(signal.SIGTERM, ternmiate_handler)



class LandmarkDataTransfer(BaseDataTransfer):


    class Item(BaseDataTransfer.BaseItem):
        MSG_TYPE = "lmk_and_disp_type"
        k_weak_num = "weak_num"
        k_image_number = "image_number"
        k_data_number = "lmk_data_number"
        k_fern_num = "fern_num"
        m_landmark_shape = "landmark_shape"
        m_landmark_dtype = "landmark_dtype"
        m_disp_shape = "disp_shape"
        m_disp_dtype = "disp_dtype"
        m_nearest_index_shape = "nearest_index_shape"
        m_nearest_dtype = "m_nearest_dtype"

        def __init__(self, uid, header):
            super().__init__(uid, header)
            # self.m_data = data
        
        def set_header(self, weak_regressor_number, fern_number):
            self.m_weak_regressor_num = weak_regressor_number
            self.m_fern_num = fern_number

            self.m_header[LandmarkDataTransfer.Item.k_weak_num]
            self.m_header[LandmarkDataTransfer.Item.k_fern_num]
            

        def set_nearest_index(self, data : np.ndarray):
            self.m_nearest_index = data 
        
            self.m_header[LandmarkDataTransfer.Item.m_nearest_index_shape] = self.m_nearest_index.shape
            self.m_header[LandmarkDataTransfer.Item.m_nearest_dtype] = self.m_nearest_index.dtype



        def set_disp(self, data : np.ndarray):
            self.m_header[LandmarkDataTransfer.Item.m_disp_shape] = data.shape
            self.m_header[LandmarkDataTransfer.Item.m_disp_dtype] = data.dtype
            self.m_disp = data 
        

        def set_landmark(self, data : np.ndarray):
            self.m_header[LandmarkDataTransfer.Item.m_landmark_shape] = data.shape
            self.m_header[LandmarkDataTransfer.Item.m_landmark_dtype] = data.dtype 

            self.m_landmark = data 
        
        
        
        def serialize(self):
            
            return self.m_header, self.m_landmark.tobytes() + self.m_disp.tobytes() + self.m_nearest_index.tobytes()
            
        

        def deserialize(self, buf):
            print(self.m_header)
            self.m_weak_regressor_num = self.m_header.get(LandmarkDataTransfer.Item.k_weak_num, 0)
            self.m_fern_num = self.m_header.get(LandmarkDataTransfer.Item.k_fern_num, 0)
            landmark_shape = self.m_header.get(LandmarkDataTransfer.Item.m_landmark_shape, -1)
            landmark_dtype = self.m_header.get(LandmarkDataTransfer.Item.m_landmark_dtype)
            disp_shape = self.m_header.get(LandmarkDataTransfer.Item.m_disp_shape, -1)
            disp_dtype = self.m_header.get(LandmarkDataTransfer.Item.m_disp_dtype)

            nearest_index_shape = self.m_header.get(LandmarkDataTransfer.Item.m_nearest_index_shape)
            nearest_index_dtype = self.m_header.get(LandmarkDataTransfer.Item.m_nearest_dtype)

            self.m_landmark =np.frombuffer(buf.read(landmark_dtype.itemsize*np.prod(landmark_shape)), landmark_dtype).reshape(landmark_shape)
            self.m_nearest_index = np.frombuffer(buf.read(np.prod(nearest_index_shape)*nearest_index_dtype.itemsize), nearest_index_dtype).reshape(nearest_index_shape)
            self.m_disp =np.frombuffer(buf.read(np.prod(disp_shape)*disp_dtype.itemsize), disp_dtype).reshape(disp_shape)

            
        def __str__(self):
            return str(self.m_nearest_index) + "\n" + str(self.m_disp) + "\n" + str(self.m_landmark)

    def __init__(self, is_writer = False):
        dir_name = osp.dirname(osp.abspath(__file__))
        super().__init__(is_writer=is_writer, python_file=osp.join(dir_name, "subprocess_helper.py"), msg_type = self.Item) 

    def recv(self):
        data = super().recv()
        return data
    


import signal 

flag = True  
def check():
    global flag 
    flag = False  
import numpy as np 
signal.signal(signal.SIGTERM, check)
if __name__ == "__main__":
    # debugpy.breakpoint()
    
    lmk_transfer = LandmarkDataTransfer()
    lmk_transfer.create()
    print("child is running")
    while flag : 
        data = lmk_transfer.recv()
        # lmk_transfer._recv() # this is ok. syncronous 
        if data is not None:
            print("aa: " ,data)            






