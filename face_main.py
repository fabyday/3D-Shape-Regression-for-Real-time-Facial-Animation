# face detection and configuration, modification.


import dlib
import cv2
import numpy as np 
import glob, copy
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


import os 
import os.path as osp



def save_lmk(path, lmks):
    root_path = osp.dirname(path)
    if not osp.exists(root_path):
        os.makedirs(root_path)
    with open(path) as fp:
        lmk_len = len(lmks)
        for i, lmk in enumerate(lmks):
            fp.write(str(lmk[0])+" "+str(lmk[1]))
            if not (lmk_len - 1 == i):
                fp.write("\n")
        
window_size =(1920, 1080)
click = False
x1, y1 = (-1, -1)
# cliped or nerest
mode = "nearest"
move = False

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
    global x1, y1, sel_rect, sel_v_idx_list, click, move, lmks, index, img_scale_factor, img_scale_factor_changed
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



cv2.namedWindow("mod")
cv2.resizeWindow('mod', *window_size)




class Image:
    def __init__(self, img, lmk):
        self.img = img 
        self.lmk = lmk

img_path = "images/"
lmk_data = [] 
img_idx = 0 
img_size = 100


index = 0
images = []
lmks = []


ext_type = [".png", ".jpeg", ".jpg"]
for ext in ext_type:
    img_names = glob.glob(osp.join(img_path, "**"+ext))
    for img_name in img_names:
        img = cv2.imread(img_name)
        images.append(img)
        lmks.append([])
img_size = len(images)

img_scale_factor = 1.0
img_scale_factor_changed = False
lmks_copy = None 
while True:  
    img_orig = cv2.cvtColor(images[index], cv2.COLOR_RGB2BGR)
    height, width = img_orig.shape[0], img_orig.shape[1]
    img = cv2.resize(img_orig, (int(width*img_scale_factor), int(height*img_scale_factor)), interpolation=cv2.INTER_LINEAR)
    if len(lmks[index]) == 0 :
        rects = detector(img, 1)
        for i, rect in enumerate(rects):
            l = rect.left()
            t = rect.top()
            b = rect.bottom()
            r = rect.right()
            shape = predictor(img, rect)
            for j in range(68):
                x, y = shape.part(j).x, shape.part(j).y
                lmks[index].append([x,y])
            lmks_copy = copy.deepcopy(lmks)
            
    
    if img_scale_factor_changed:
        for i, lmk in enumerate(lmks[index]):
            lmk[0]=int(lmks_copy[index][i][0]*img_scale_factor)
            lmk[1]=int(lmks_copy[index][i][1]*img_scale_factor)
        img_scale_factor_changed = False


    key = cv2.waitKey(1)
    if key == ord('q'): #quit
        break
    elif key == ord("d"): # move next
        index += 1
        if index > img_size - 1 :
            index = 0
    elif key == ord("a"): # move prev
        index -= 1
        if index < 0:
            index = img_size - 1
    elif key == ord("s"): # save current images.
        pass
    elif key == ord("r"): # move true
        move = not move

    for j in range(68):
        x, y = lmks[index][j]
        color = (0,255,0)
        if j in sel_v_idx_list:
            color = (255,0,0)
        cv2.circle(img, (x, y), 1, color, int(10*img_scale_factor))

    if not sel_rect == [-1, -1, -1, -1]:
        cv2.rectangle(img, sel_rect[:2], sel_rect[2:], color=(255,255,0), thickness=int(10*img_scale_factor))
    cv2.imshow('mod', img)
    cv2.setMouseCallback("mod", mouse_event, None)


cv2.destroyAllWindows()
