# face detection and configuration, modification.


import dlib
import cv2
import numpy as np 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")






def save_lmk():
    pass

click = False
x1, y1 = (-1, -1)
# cliped or nerest
mode = "nearest"


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
    nearest_pts_idx = -1
    shortest_length = np.inf
    for i, v in enumerate(v_list) : 
        length = np.sqrt((v[0] - x)**2 + (v[1]-y)**2)
        if length < eps and length < shortest_length:
            shortest_length = length
            nearest_pts_idx = i
    return nearest_pts_idx


def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:                      # 마우스를 누른 상태
        click = True 
        x1, y1 = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if click and sel_v_idx_list: 
            m_vec_x, m_vec_y = x - x1, y - y1
            for idx in sel_v_idx_list:
                v_list[idx][0] += m_vec_x 
                v_list[idx][1] += m_vec_y
    elif event == cv2.EVENT_LBUTTONUP:
        click = False 
        if mode == "cliped":
            sel_v_idx_list = find_v_in_rect(x1, y1, x, y)
            sel_rect = [x1, y1, x, y]
        elif mode == "nearest":
            sel_v_idx_list = [find_v_in_nearest_area(x1, y1, eps = 10)]

    

cv2.setMouseCallback("mod", mouse_event)


lmk_data = [] 
img_idx = 0 
img_size = 100
while True:  
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # r = 200. / img.shape[1]
    # dim = (200, int(img.shape[0] * r))    
    # resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    rects = detector(img, 1)
    for i, rect in enumerate(rects):
        l = rect.left()
        t = rect.top()
        b = rect.bottom()
        r = rect.right()
        shape = predictor(img, rect)
        for j in range(68):
            x, y = shape.part(j).x, shape.part(j).y
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 2)
        cv2.imshow('mod', img)
        
        key = cv2.waitKey(1)
        
        if key == ord('q'):
            break
        elif key == ord("d"):
            
        elif key == ord("a"):
            pass

cv2.destroyAllWindows()
