
# import dlib
import cv2
import numpy as np 
import glob, copy


import os 
import os.path as osp
import data_loader as dl 

def save_lmk(path, lmk):
    root_path = osp.dirname(path)
    print("root", root_path)
    if not osp.exists(root_path):
        os.makedirs(root_path)
    with open(path, 'w') as fp:
        lmk_len = len(lmk)
        for i, coord in enumerate(lmk):
            fp.write(str(coord[0])+" "+str(coord[1]))
            if not (lmk_len - 1 == i):
                fp.write("\n")
        
window_size =(1920, 1080)
click = False
x1, y1 = (-1, -1)
# cliped or nerest
mode = "nearest"
move = False

is_moved_point = False

sel_v_idx_list = []
sel_rect = [-1,-1,-1,-1]

img = None 
v_list = [] 
def find_v_in_rect(x1, y1, x2, y2):
    min_x = min(x1, x2)
    min_y = min(y1, y2)
    max_x = max(x1, x2)
    max_y = max(y1, y2)
    ind = []
    for i, v in enumerate(v_list) : 
        if (min_x < v[0] and max_x > v[0]) and (min_y < v[1] and max_y > v[1]):
            ind.append(i)




class Image:
    def __init__(self, parent, img_path, lmk_size, category=None, lazy_load = False):
        self.img_path = img_path

        self.parent = parent
        self.lmk = [[0,0] for _ in range(lmk_size)]
        
        self.img_name = osp.splitext(osp.basename(img_path))
        self.category = category
        self.img = None
        
        self.img_scale_factor  = 1.0
        
        self.lmk_detected = False
        
        if not lazy_load:
            self.load()
    
    def is_landmark_detected(self):
        return self.lmk_detected
    
    
    def load(self, force_reload = False):
        if force_reload:
            self.img = None

        if not self.is_loaded():
            self.img = cv2.imread(self.img_path)
            return True
        return False

    def is_loaded(self):
        return False if self.img is None else True
    
    def size(self):
        height, width = self.img.shape[0], self.img.shape[1]
        return width, height

    def set_lmk(self, index, x, y):
        self.lmk[index][0] = x 
        self.lmk[index][1] = y
    
    def get_image(self):
        return self.img

    def get_lmk(self):
        copied_lmk = copy.deepcopy(self.lmk)
        return copied_lmk

    
    def __str__(self):
        return "name : " + self.img_name + " category : " + (self.category if self.category != None else "???" )

class ImageCollectionIterator:
    def __init__(self, collection):
        self.collection = collection
        self.index_list = []

        self.predictor = None
        tmp = copy.deepcopy(list(self.collection.meta.items()))
        tmp.reverse()
        self.item_stack = tmp
        self.key_stack = []
    
    def __iter__(self):
        return self 
    
    def __next__(self):
        while len(self.item_stack) != 0 :
            key, item = self.item_stack.pop()
            if isinstance(item, dict):
                for item_key in item.keys():
                    self.item_stack.append(([key, item_key], item[item_key]))
            elif isinstance(item, list):
                for name in item:
                    self.item_stack.append((key, name))
            else :
                return self.collection[[key, item]] # item must be string.
        
        raise StopIteration 

class ImageCollection():
    def __init__(self, image_dir, lmk_meta_file, lazy_load = False):
        self.meta, self.ext = dl.load_image_meta(osp.join(image_dir, "meta.yaml"))
        self.lazy_load = lazy_load

        self.full_index, self.eye, self.contour, self.mouse, self.eyebrow, self.nose = dl.load_ict_landmark(lmk_meta_file)
        self.lmk_size = len(self.full_index)

        self.img_data_infos = dict()
        self.img_data_list = []
        for expr_key in self.meta.keys():
            self.img_data_infos[expr_key] = dict()
            expr_image_collection = self.img_data_infos[expr_key]
            for img_key in self.meta[expr_key]:
                img_path = osp.join(image_dir, img_key+self.ext)
                expr_image_collection[img_key] = Image(self, img_path, self.lmk_size, img_key, expr_key, lazy_load=lazy_load)
                self.img_data_list.append(expr_image_collection[img_key])

    
    def load_predictor(self, path=None):
        import dlib
        if path == None :
            path = "./shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(path)

    def get_detector_and_predictor(self):
        return self.detector, self.predictor

    def __len__(self):
        return len(self.img_data_list)

    def __iter__(self):
        return ImageCollectionIterator(self)

    def __getitem__(self, name):
        if isinstance(name , int):
            return self.img_data_list[name]

        if isinstance(name, str):
            return self.img_data_infos[name]

        reval = self.img_data_infos
        for key in name:
            reval = reval[key]
        return reval
        


    



