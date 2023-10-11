import cv2 
import numpy as np 
import copy 



class HalfEdge:
    def __init__(self, face_index_list = None):
        v_e_list = dict() 
        edge_idx = []
        face_index = face_index_list

    
    def build(face_index_list = None ):
        if face_index_list != None :
            face_index = face_index_list


        
        
        if face_index == None :
            raise ValueError("face index is None")
        
    def is_boundrary():
        pass 

    def vv(v):
        pass 

    def ff(f1, f2):
        pass


def get_half_edge(index_list):
    v = dict()
    max_idx = 0
    #find max index
    for faces in index_list:
        for f in faces:
            if max_idx < f:
                max_idx = f
    

    for i in range(max_idx+1):
        v[i] = set()
    



def resize(img, width):
    h,w,c = img.shape
    ratio = width / w 
    new_h = h*ratio 
    img = cv2.resize(img, [int(new_h), int(width)])
    return img


def find_boundary_pts(idx_list): # this is not water tight
    res = dict()
    max_idx = 0
    #find max index
    for faces in idx_list:
        for f in faces:
            if max_idx < f:
                max_idx = f
    

    for i in range(max_idx+1):
        res[i] = set()
    

    for f1, f2, f3 in idx_list:
        res[f1].add((f1,f3)); res[f1].add((f1,f2))
        res[f2].add((f2,f3)); res[f2].add((f2,f1))
        res[f3].add((f3,f1)); res[f3].add((f3,f2))
    
    #find boundrary
    boundary_idx = []
    for key in res.keys():
        if len(res[key])  < 2:
            boundary_idx.append(key)
    
    if len(boundary_idx) == 0 :
        return boundary_idx
    
    #find connection.
    started_b_idx = boundary_idx[0]
    cur_b_idx = started_b_idx
    sorted_boundary_idx = [cur_b_idx]
    while True:
        for tmp_b_idx in res[cur_b_idx]:
            if tmp_b_idx in boundary_idx:
                sorted_boundary_idx.append(tmp_b_idx)
                cur_b_idx = tmp_b_idx
        
        if cur_b_idx == started_b_idx:
            break 
    return sorted_boundary_idx

def draw_circle(v, img, colors = (1.0,0.0,0.0)):
    for vv in v:
        cv2.circle(img, center=vv.astype(int), radius=10, color=colors, thickness=2)


def draw_contour(img, lmk, new_contour, color = (0,0,255), line_color =(255,0,0), width = None ):
    cp_img = copy.deepcopy(img)

    draw_circle(lmk,cp_img, colors=color)
    for i in range(len(new_contour) -1):
        cv2.line(cp_img, new_contour[i].astype(int), new_contour[i+1].astype(int), color=line_color, thickness=3)
    
    if width == None :
        resized_img = cp_img
    else:
        resized_img = resize(cp_img, width)


    return resized_img


    for i in range(len(orig) -1):
        cv2.line(img, orig[i].astype(int), orig[i+1].astype(int), color=(255,255,0), thickness=3)
    img= resize(img, 1500)

def concatenate_img(*imgs):
    show_img = np.concatenate(imgs, 1)
    return show_img


def draw_pts(img, pts_2d, color = (0,0,255), width = None):
    
    cp_img = copy.deepcopy(img)
    
    draw_circle(pts_2d, cp_img, color)
    if width == None :
        resized_img = cp_img
    else:
        resized_img = resize(cp_img, width)
    return resized_img
