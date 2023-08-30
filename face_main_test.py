# face detection and configuration, modification.


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

def find_v_in_nearest_area(x,y, eps = 1.5):
    # eps sqrt(2) < 1.5 .. this is min size of pixel(cross)
    global lmks, index
    v_list = lmks
    nearest_pts_idx = -1
    shortest_length = np.inf
    for i, v in enumerate(v_list[index]) : 
        length = np.sqrt((v[0] - x)**2 + (v[1]-y)**2)
        if length < eps and length < shortest_length:
            shortest_length = length
            nearest_pts_idx = i
    return nearest_pts_idx


def mouse_event(event, x, y, flags, param):
    global x1, y1, sel_rect, sel_v_idx_list, click, move, lmks, index, img_scale_factor, img_scale_factor_changed, is_moved_point
    if event == cv2.EVENT_LBUTTONDOWN:                      
        click = True 
        x1, y1 = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if click and move and sel_v_idx_list:

            m_vec_x, m_vec_y = x - x1, y - y1
            x1 = x; y1 = y
            for idx in sel_v_idx_list:
                lmks[index][idx][0] += m_vec_x 
                lmks[index][idx][1] += m_vec_y
            is_moved_point = True
    elif event == cv2.EVENT_LBUTTONUP:
        click = False 
        if mode == "cliped":
            sel_v_idx_list = find_v_in_rect(x1, y1, x, y)
            sel_rect = [x1, y1, x, y]
        elif mode == "nearest":
            sel_v_idx_list = [find_v_in_nearest_area(x1, y1, eps = 200)]

    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags >0: # scroll up
            img_scale_factor += 0.1
        else: # scroll down
            img_scale_factor -= 0.1
        img_scale_factor_changed = True



class LandmarkMeta:
    def __init__(self):
        pass


class Image:
    def __init__(self, img_path, lmk_size, img_name, category=None, lazy_load = False):
        self.img_path = img_path
        self.lmk = [[0,0] for _ in range(lmk_size)]
        self.img_name = img_name
        self.category = category
        self.img = None
        
        self.img_scale_factor  = 1.0
        
        self.lmk_detected = False
        self.redetect_flag = False
        if not lazy_load:
            self.load()
    
    def is_landmark_detected(self):
        return self.lmk_detected
    def calc_lankmark_from_dlib(self, detector, predictor, redetect=False):
        import dlib
        self.load()
        redetect = self.redetect_flag 
        self.redetect_flag = False
        img = self.img
        h,w, _ = img.shape
        max_length = max(h, w)
        if max_length > 1000:
            max_length_ratio = max_length / 1000
        else :
            max_length_ratio = 1

        img = cv2.resize(img, [int(w/max_length_ratio),int(h/max_length_ratio)])
        if not self.is_landmark_detected() or redetect:
            rects = detector(img, 1)
            for i, rect in enumerate(rects):
                l = rect.left()
                t = rect.top()
                b = rect.bottom()
                r = rect.right()
                shape = predictor(img, rect)
                for j in range(68):
                    x, y = shape.part(j).x, shape.part(j).y
                    self.lmk[j][0] = x*max_length_ratio
                    self.lmk[j][1] = y*max_length_ratio
            self.lmk_detected = True
            return True
        return False
    
    def load(self):
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
        height, width = self.img.shape[0], self.img.shape[1]
        return cv2.resize(self.img, (int(width*img_scale_factor), int(height*img_scale_factor)), interpolation=cv2.INTER_LINEAR)

    def get_lmk(self):
        copied_lmk = copy.deepcopy(self.lmk)
        for lmk in copied_lmk:
            lmk[0] *= img_scale_factor
            lmk[1] *= img_scale_factor
        return copied_lmk

    # lmk, img is influenced by scale factor
    def set_scale_factor(self, img_scale_factor):
        self.img_scale_factor = img_scale_factor
        
    
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

        self.full_index, self.eye, self.contour, self.mouse, self.eyebrow = dl.load_ict_landmark(lmk_meta_file)
        self.lmk_size = len(self.full_index)

        self.img_data_infos = dict()
        self.img_data_list = []
        for expr_key in self.meta.keys():
            self.img_data_infos[expr_key] = dict()
            expr_image_collection = self.img_data_infos[expr_key]
            for img_key in self.meta[expr_key]:
                img_path = osp.join(image_dir, img_key+self.ext)
                expr_image_collection[img_key] = Image(img_path, self.lmk_size, img_key, expr_key, lazy_load=lazy_load)
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
        


    
# img_collection = ImageCollection("./images/all_in_one/expression", "./ict_lmk_info.yaml")





