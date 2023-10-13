import cv2 
import numpy as np 
import copy 



class EdgeFace:
    def __init__(self, face_index_list = None):
        self.edge_face = {}
        self.vert_edge = []
        self.vert_face = []
        self.edge = []
        self.face_index = face_index_list
        self.face_edge = []

        self.is_built = False
    def minmax(self, a,b):
        return (min(a,b), max(a,b))
    
    def build(self, face_index_list = None ):
        if face_index_list != None :
            self.face_index = face_index_list

        max_idx = -1
        for faces in self.face_index:
            for f in faces:
                if max_idx < f:
                    max_idx = f
        
        self.vert_edge = [ set() for _ in range(max_idx + 1) ]
        self.vert_face = [set() for _ in range(max_idx + 1)]

        if self.face_index is None :
            raise ValueError("face index is None")
        self.face_edge = [[] for _ in range(len(self.face_index))]        
        for fi, (v1,v2,v3 ) in enumerate(self.face_index):
            edge1 = self.minmax(v1,v2)
            edge2 = self.minmax(v2,v3)
            edge3 = self.minmax(v3,v1)
            
            edge1_idx = len(self.edge)
            self.edge.append(edge1)
            edge2_idx = len(self.edge)
            self.edge.append(edge2)
            edge3_idx = len(self.edge)
            self.edge.append(edge3)

            self.vert_edge[v1].add(edge1_idx);self.vert_edge[v1].add(edge3_idx)
            self.vert_edge[v2].add(edge1_idx);self.vert_edge[v2].add(edge2_idx)
            self.vert_edge[v3].add(edge2_idx);self.vert_edge[v3].add(edge3_idx)
            self.vert_face[v1].add(fi)
            self.vert_face[v2].add(fi)
            self.vert_face[v2].add(fi)

            self.face_edge[fi].append(edge1_idx)
            self.face_edge[fi].append(edge2_idx)
            self.face_edge[fi].append(edge3_idx)
            
            for edge_pair in [edge1, edge2, edge3]:
                if self.edge_face.get(edge_pair, None) is None :
                    self.edge_face[edge_pair] = []
            self.edge_face[edge1].append(fi)
            self.edge_face[edge2].append(fi)
            self.edge_face[edge3].append(fi)
        


        test_list = []
        for k in self.edge_face:
            if len(self.edge_face[k]) < 2 : 
                test_list += self.edge_face[k]
        print("")
        # boundary may not be only one.
        ####################################################
        
        #find candidate boundary face and edge
        candidate_face_index_list = []
        candidate_edge_index_list = []
        test2 = []
        for f_idx in range(len(self.face_index)):
            if self.is_boundrary(f_idx):
                test2.append(f_idx)
                candidate_face_index_list.append(f_idx)
                candidate_edge_index_list.append(self.get_boundary_edge(f_idx))
        print(set(test2) - set(test_list))
        # linked boundary index.
        self.boundary_v_index_list = []
        


        # find looooooooooooooooooooooooooop
        self.boundary_v_idx_list = []
        self.boundary_edge_idx_list = []
        consumed_edge_set = set()
        for e_idx in candidate_edge_index_list:

            if e_idx in  consumed_edge_set : 
                print("this was already used.")
                continue 

            boundary_edge_idx_list = []
            boundary_v_idx = []
            next_edge_idx = e_idx
            while True:
                edge = self.edge[next_edge_idx]
                if len(boundary_edge_idx_list) != 0:
                    if next_edge_idx == e_idx: # we finally find loop of this edges.
                        break
                    prev_edge = self.edge[ boundary_edge_idx_list[-1] ] 
                    v2 = list(set(edge) - set(prev_edge))[0]
                    boundary_edge_idx_list.append(next_edge_idx)
                else: # when start to find edges.
                    boundary_edge_idx_list.append(next_edge_idx)
                    boundary_v_idx.append(edge[0])
                    v2 = edge[1]

                boundary_v_idx.append(v2)
                edges = list(self.vert_edge[v2])
                boundary_edge_checker = [self.is_boundrary_edge(edge_idx) for edge_idx in edges]

                prev_idx = next_edge_idx
                next_edge_idx = None
                for i, is_boundary in enumerate(boundary_edge_checker): 
                    if is_boundary and prev_idx != edges[i]:
                        next_edge_idx = edges[i]
                
                if next_edge_idx == None :
                    raise ValueError("boundary edge was not connected to this edge.")
            self.boundary_v_idx_list.append(boundary_v_idx)
            consumed_edge_set = consumed_edge_set.union(boundary_edge_idx_list)
            self.boundary_edge_idx_list.append(boundary_edge_idx_list)
        
        
        self.is_built = True


    def get_boundary_v_list(self):
        if not self.is_built : 
            raise RuntimeError("this was not built")
        return self.boundary_v_idx_list

    def get_boundary_edge(self, face_index):
        edge_indices = self.face_edge[face_index]
        for ed_idx in edge_indices:
            edge_vv_key = self.edge[ed_idx]
            if len(self.edge_face[edge_vv_key]) < 2:
                return ed_idx
        
    def is_boundrary(self,face_index):
        edge_indices = self.face_edge[face_index]
        boundary_bool = False
        for ed_idx in edge_indices:
            edge_vv_key = self.edge[ed_idx]
            if len(self.edge_face[edge_vv_key]) < 2:
                boundary_bool = True
                break

        return boundary_bool
    

    def is_boundrary_edge(self, edge_idx):
        boundary_bool = False
        edge_vv_key = self.edge[edge_idx]
        if len(self.edge_face[edge_vv_key]) < 2:
            boundary_bool = True

        return boundary_bool

    def vv(v):
        pass 

    def ff(f1, f2):
        pass




class CvKey:


    def __init__(self):

        self.delay = 0

    def load_config(self, file_name):
        pass 


    def set_delay(self, delay):
        self.delay = delay





g_key_manager = CvKey()



def resize(img, width):
    w, h, c = img.shape
    if w == width:
        return img
    ratio = width / w 
    new_h = h*ratio 
    img = cv2.resize(img, [int(new_h), int(width)])
    return img


def put_text(img, caption, color, base_loc, font_face, font_scale, thickness, linetype, fit_bool = False):
    t_size = cv2.getTextSize(caption, 2, fontScale=font_scale, thickness=thickness+2)
    base_loc_x, base_loc_y = base_loc
    wid = t_size[0][0]
    hei = t_size[0][1]
    cv2.putText(img, caption, (base_loc_x , base_loc_y+hei+2), font_face, font_scale, (255, 255,255), thickness+3, cv2.LINE_AA) # white outline
    cv2.putText(img, caption, (base_loc_x , base_loc_y+hei+2), font_face, font_scale, (0,0,0), thickness+1, cv2.LINE_AA) # black outline
    cv2.putText(img, caption, (base_loc_x , base_loc_y+hei+2), font_face, font_scale, color, thickness, cv2.LINE_AA)


def draw_circle(v, img, colors = (1.0,0.0,0.0), radius = 10):
    for vv in v:
        cv2.circle(img, center=vv.astype(int), radius=radius, color=colors, thickness=2)


def draw_contour(img, lmk, new_contour, color = (0,0,255), line_color =(255,0,0), width = None, caption = "" ):
    cp_img = copy.deepcopy(img)
    
    contour = new_contour
    sel_lmk = lmk[contour, :].astype(int)
    
    draw_circle(sel_lmk,cp_img, colors=color)
    for i in range(len(new_contour) -1):
        cv2.line(cp_img, sel_lmk[i], sel_lmk[i+1], color=line_color, thickness=3)
    
    if width == None :
        resized_img = cp_img
    else:
        resized_img = resize(cp_img, width)
    caption_size = len(caption)
    put_text(resized_img, caption, color, base_loc=(0,0), \
             font_face = 1, font_scale=2, thickness=1, linetype=cv2.LINE_AA, fit_bool=False )
    return resized_img

def concatenate_img(row, col,*imgs):
    if len(imgs) != row*col:
        raise ValueError("row and col is not same size as imgs")
    cols = []
    for j in range(len(imgs)//col):
        cols.append(np.concatenate(imgs[col*j:col*(j+1)],1))

        
    show_img = np.concatenate(cols, 0)
    return show_img


def draw_pts(img, pts_2d, color = (0,0,255), width = None, radius = 10, caption=""):
    
    cp_img = copy.deepcopy(img)
    
    draw_circle(pts_2d, cp_img, color, radius)
    if width == None :
        resized_img = cp_img
    else:
        resized_img = resize(cp_img, width)
    return resized_img


def set_delay(delay):
    g_key_manager.set_delay(delay)

def show(title, img):
    cv2.imshow(title, img)
    key = cv2.waitKey(g_key_manager.delay)
    if key == ord('a'):
        return
    elif key == ord('w'):
        return
    elif key == ord('d'):
        return
    elif key == ord('s'):
        return 
    elif key == ord('q'):
        exit(0)