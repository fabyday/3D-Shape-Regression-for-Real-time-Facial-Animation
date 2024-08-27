import cv2 
import numpy as np 
import copy 
import glm


RED = (0,0,255)
BLUE = (255,0,0)
GREEN = (0,255,0)
YELLO = (0,255,255)
SKY_BLUE = (255,255,0)



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
                # print("this was already used.")
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



def resize_img(img, width):
    if len(img.shape) == 3:
        w, h, c = img.shape
    else :
        w, h = img.shape
    if w == width:
        return img
    ratio = width / w 
    new_h = h*ratio 
    img = cv2.resize(img, [int(new_h), int(width)])
    return img

def put_text(img, caption, color, base_loc=(0,0), font_face = 1, font_scale=2, thickness=1, linetype=cv2.LINE_AA, fit_bool=False):
    t_size = cv2.getTextSize(caption, 2, fontScale=font_scale, thickness=thickness+2)
    base_loc_x, base_loc_y = base_loc
    wid = t_size[0][0]
    hei = t_size[0][1]
    cv2.putText(img, caption, (base_loc_x , base_loc_y+hei+2), font_face, font_scale, (255, 255,255), thickness+3, cv2.LINE_AA) # white outline
    cv2.putText(img, caption, (base_loc_x , base_loc_y+hei+2), font_face, font_scale, (0,0,0), thickness+1, cv2.LINE_AA) # black outline
    cv2.putText(img, caption, (base_loc_x , base_loc_y+hei+2), font_face, font_scale, color, thickness, cv2.LINE_AA)


def draw_circle(v, img, colors = (1.0,0.0,0.0), radius = 10, thickness=2):
    img = np.copy(img)
    for vv in v:
        cv2.circle(img, center=vv.astype(int), radius=radius, color=colors, thickness=thickness)
        
    return img

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
        resized_img = resize_img(cp_img, width)
    caption_size = len(caption)
    put_text(resized_img, caption, color, base_loc=(0,0), \
             font_face = 1, font_scale=2, thickness=1, linetype=cv2.LINE_AA, fit_bool=False )
    return resized_img

def concatenate_img(row, col,*imgs):
    if len(imgs) > row*col:
        raise ValueError("row and col is not same size as imgs")
    cols = []




    last_one = len(imgs)
    


    for i in range(row):
        same_line_img = [] 
        for j in range(col):
            idx = i*row + j
            if idx < last_one :
                same_line_img.append(imgs[idx])
            else :
                same_line_img.append(np.zeros_like(imgs[0]))
        concat_img = np.concatenate(same_line_img, 1)
        cols.append(concat_img)
    show_img = np.concatenate(cols, 0)
    
        
    return show_img

def draw_pts_mapping(img, pts1_list, pts2_list, color = (0,0,255), width = None, thickness = 3, caption=""):
    cp_img = copy.deepcopy(img)
    
    

    for pts1, pts2 in zip(pts1_list, pts2_list):
        cv2.line(cp_img, pts1.astype(np.int32), pts2.astype(np.int32), color=color, thickness=thickness)

    if width == None :
        resized_img = cp_img
    else:
        resized_img = resize_img(cp_img, width)
    put_text(resized_img, caption, color, base_loc=(0,0), \
            font_face = 1, font_scale=2, thickness=1, linetype=cv2.LINE_AA, fit_bool=False )
    return resized_img

def draw_pts(img, pts_2d, color = (0,0,255), width = None, radius = 10, thickness = 2, caption=""):
    
    cp_img = copy.deepcopy(img)
    
    cp_img = draw_circle(pts_2d, cp_img, color, radius,thickness=thickness)
    if width == None :
        resized_img = cp_img
    else:
        resized_img = resize_img(cp_img, width)
    put_text(resized_img, caption, color, base_loc=(0,0), \
            font_face = 1, font_scale=2, thickness=1, linetype=cv2.LINE_AA, fit_bool=False )
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


from OpenGL.WGL import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import cv2
import uuid
from win32api import *
from win32con import *
from win32gui import *

PFD_TYPE_RGBA =         0
PFD_MAIN_PLANE =        0
PFD_DOUBLEBUFFER =      0x00000001
PFD_DRAW_TO_WINDOW =    0x00000004
PFD_SUPPORT_OPENGL =    0x00000020
def mywglCreateContext(hWnd):
    pfd = PIXELFORMATDESCRIPTOR()

    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL
    pfd.iPixelType = PFD_TYPE_RGBA
    pfd.cColorBits = 32
    pfd.cDepthBits = 24
    pfd.iLayerType = PFD_MAIN_PLANE

    hdc = GetDC(hWnd)

    pixelformat = ChoosePixelFormat(hdc, pfd)
    SetPixelFormat(hdc, pixelformat, pfd)

    oglrc = wglCreateContext(hdc)
    wglMakeCurrent(hdc, oglrc)

    # check is context created succesfully
    # print "OpenGL version:", glGetString(GL_VERSION)

class MeshWriter:
    def __init__(self):
        self._init_context()
        self._init_shader_and_program()
        self._init_buffer()
        self.fbo = None
        self.color_buf = None
        self.depth_buf = None
        self.set_light_color(1.0,1.0,1.0)
        self.set_light_pos(0.0,0.0,0.0)
        self.width = 0    
        self.height = 0    


    def set_light_pos(self, x,y,z):
        self.light_pos = [x,y,z]
        
    
    def set_light_color(self, r, g ,b):
        self.bgr = [b,g,r]
        



    def _init_context(self):
        # see also https://stackoverflow.com/questions/41126090/how-to-write-pyopengl-in-to-jpg-image
        hInstance = GetModuleHandle(None)

        wndClass = WNDCLASS()

        wndClass.lpfnWndProc = DefWindowProc
        wndClass.hInstance = hInstance
        wndClass.hbrBackground = GetStockObject(WHITE_BRUSH)
        wndClass.hCursor = LoadCursor(0, IDC_ARROW)
        wndClass.lpszClassName = str(uuid.uuid4())
        wndClass.style = CS_OWNDC

        wndClassAtom = RegisterClass(wndClass)

        # don't care about window size, couse we will create independent buffers
        self.hWnd = CreateWindow(wndClassAtom, '', WS_POPUP, 0, 0, 1, 1, 0, 0, hInstance, None)

        # Ok, window created, now we can create OpenGL context

        mywglCreateContext(self.hWnd)

    def _init_shader_and_program(self):
        gl_program = glCreateProgram()
        v_shader = glCreateShader(GL_VERTEX_SHADER)
        p_shader = glCreateShader(GL_FRAGMENT_SHADER)

        v_shader_src = """
            #version 330 core
            layout (location = 0) in vec3 aPos;
            layout (location = 1) in vec3 anormal;

            uniform mat4 Q;
            uniform mat4 Rt;

            out vec3 Normal;
            out vec3 FragPos;

            void main()
            {
                gl_Position = Q*Rt*vec4(aPos, 1.0);
                //FragPos = vec3(Rt*vec4(aPos,1.0));
                //Normal = mat3(transpose(inverse(Rt)))*anormal;
                FragPos =aPos;
                Normal = anormal;
            }
            """

        p_shader_src ="""
            #version 330 core
            out vec4 FragColor;

            uniform vec3 lightColor;
            uniform vec3 objectColor;
            uniform vec3 lightPos;

            in vec3 Normal;
            in vec3 FragPos;


            void main()
            {
                float ambientStrength = 0.1;
                vec3 ambient = ambientStrength * (lightColor);
                vec3 norm = normalize(Normal);
                vec3 lightDir = normalize(lightPos - FragPos);
                
                float diff = max(dot(norm, lightDir), 0.0);
                vec3 diffuse = diff*lightColor;
                vec3 result = (ambient + diffuse)*objectColor;
                FragColor = vec4(result, 0.0);
            }
            """

        glShaderSource(v_shader, v_shader_src)
        glShaderSource(p_shader, p_shader_src)


        glCompileShader(v_shader)
        if not glGetShaderiv(v_shader, GL_COMPILE_STATUS):
            info_log = glGetShaderInfoLog(v_shader)
            print ("v shader Compilation Failure for " + str(v_shader) + " shader:\n" + str(info_log))
        glCompileShader(p_shader)
        if not glGetShaderiv(p_shader, GL_COMPILE_STATUS):
            info_log = glGetShaderInfoLog(p_shader)
            print ("p shader Compilation Failure for " + str(p_shader) + " shader:\n" + str(info_log))

        glAttachShader(gl_program, v_shader)
        glAttachShader(gl_program, p_shader)

        glLinkProgram(gl_program)
        glDeleteShader(v_shader)
        glDeleteShader(p_shader)    
        
        status = glGetProgramiv(gl_program, GL_LINK_STATUS)
        if status :
            test = glGetProgramInfoLog(gl_program)
            print(test)

        self.gl_program = gl_program

 
    
    def deinit_offscreen_renderer(self):
        if self.fbo is None :
            return 
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDeleteRenderbuffers(1, self.color_buf)
        glDeleteRenderbuffers(1, self.depth_buf)
        glDeleteFramebuffers(1, self.fbo)


        self.fbo = None 
        self.color_buf = None 
        self.depth_buf = None 

    def init_off_screen_renderer(self):
        self.fbo = glGenFramebuffers(1)
        self.color_buf = glGenRenderbuffers(1)
        self.depth_buf = glGenRenderbuffers(1)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)



        glBindRenderbuffer(GL_RENDERBUFFER, self.color_buf)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, self.width, self.height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self.color_buf)


        # bind depth render buffer - no need for 2D, but necessary for real 3D rendering
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_buf)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.width, self.height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth_buf)

    def _init_buffer(self):
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)



    def set_render_img_size(self, width, height):
        if self.width == width and self.height == height and self.fbo is not None:
            return
        self.width = width
        self.height = height
        self.deinit_offscreen_renderer()
        self.init_off_screen_renderer()

    def read_color_buffer(self):
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        res = np.zeros((self.width*self.height*4), dtype=np.uint8)
        
        glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, res.ctypes.data_as(ctypes.c_void_p))
        res = res.reshape(self.height, self.width, -1)
        return res

    def convert_hz_intrinsic_to_opengl_projection(self, K,x0,y0,width,height,znear,zfar, window_coords=None):
        #https://gist.github.com/astraw/1341472#file_calib_test_numpy.py
        #https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/
        znear = float(znear)
        zfar = float(zfar)
        depth = zfar - znear
        q = -(zfar + znear) / depth
        qn = -2 * (zfar * znear) / depth

        if window_coords=='y up':
            proj = np.array([[ 2*K[0,0]/width, -2*K[0,1]/width, (-2*K[0,2]+width+2*x0)/width, 0 ],
                            [  0,             -2*K[1,1]/height,(-2*K[1,2]+height+2*y0)/height, 0],
                            [0,0,q,qn],  # This row is standard glPerspective and sets near and far planes.
                            [0,0,-1,0]]) # This row is also standard glPerspective.
        else:
            assert window_coords=='y down'
            proj = np.array([[ 2*K[0,0]/width, -2*K[0,1]/width, (-2*K[0,2]+width+2*x0)/width, 0 ],
                            [  0,              2*K[1,1]/height,( 2*K[1,2]-height+2*y0)/height, 0],
                            [0,0,q,qn],  # This row is standard glPerspective and sets near and far planes.
                            [0,0,-1,0]]) # This row is also standard glPerspective.
        return proj

    def draw(self, img, Q, Rt, v, f, v_normal, color):
        if isinstance(color, list) or isinstance(color, tuple):
            color = np.array(color, dtype=np.float32)
        h, w, _ = img.shape
        self.set_render_img_size(w, h)

        flat_vnv = np.concatenate([v, v_normal], axis=-1).reshape(-1).astype(np.float32)
        flat_tmp = np.zeros_like(flat_vnv, dtype=np.float32)
        flat_tmp[...] = flat_vnv 
        flat_vnv = flat_tmp
        flat_f = np.zeros((f.size), dtype=np.uint32)
        flat_f[...] = f.reshape(-1).astype(np.uint32)
        glBufferData(GL_ARRAY_BUFFER, flat_vnv.itemsize*flat_vnv.size, flat_vnv.ctypes.data_as(ctypes.c_void_p), GL_STATIC_DRAW)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, flat_f.itemsize*flat_f.size, flat_f.ctypes.data_as(ctypes.c_void_p), GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, flat_vnv.itemsize*6, ctypes.c_void_p(0))
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, flat_vnv.itemsize*6, ctypes.c_void_p( 3 * ctypes.sizeof(ctypes.c_float) ) )
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        # glPolygonMode( GL_FRONT_AND_BACK, GL_LINE )
        glFrontFace(GL_CW)
        glPolygonMode( GL_FRONT, GL_FILL )
        # glPolygonMode( GL_BACK, GL_EMP )
        # glDisable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glEnable(GL_CULL_FACE)
        
        glEnable(GL_DEPTH_TEST);  
        glEnable(GL_DEPTH_CLAMP)
        glDepthFunc(GL_LESS)
        glViewport(0,0, self.width, self.height)
        glUseProgram(self.gl_program)
        # glCullFace()
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        new_Rt = np.identity(4, dtype=np.float32)
        # https://amytabb.com/tips/tutorials/2019/06/28/OpenCV-to-OpenGL-tutorial-essentials/
        new_Rt[:3,:3] = Rt[:3,:3]
        new_Rt[:-1, -1] = Rt[:, -1]

        near = 0.1
        far = 1000.0
        
        
        


        # new_NDC[0,0] = -2/self.width; new_NDC[0,-1] =1
        # new_NDC[1,1] = 2/self.height; new_NDC[1,-1] = -1
        # new_NDC[2,2] = -2/(far-near); new_NDC[2,-1] = -(far+near)/(far-near)
        # new_NDC[3,-1] = 1
        def gen_proj(Q, left, right, bottom, top, near, far):
            near = near 
            far = far 
            Q = np.copy(Q)
            Q /= Q[-1,-1] # remove alpha
            # see https://sightations.wordpress.com/2010/08/03/simulating-calibrated-cameras-in-opengl/
            #see http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
            #see http://jamesgregson.blogspot.com/2011/11/matching-calibrated-cameras-with-opengl.html
            near = near 
            far = far 
            Q = np.copy(Q)
            Q /= Q[-1,-1] # remove alpha

            alpha = Q[0,0]
            beta = Q[1,1]
            skew = Q[0,1]
            u0 = Q[0,2]
            v0 = Q[1,2]
            
            proj = np.zeros((4,4), dtype=np.float32)
            ndc = np.zeros((4,4), dtype=np.float32)

            ndc[0,0] = 2.0/(right - left) ; ndc[0,2] = -(right + left)/(right-left)
            ndc[1,1] = 2.0/(top - bottom) ; ndc[1,2] = -(top + bottom)/(top - bottom)
            ndc[2,2] = -(far+near)/(far - near) ; ndc[2,2] = -(far * near)/(far - near)
            # ndc[2,2] = (far+near)/(far - near) ; ndc[2,2] = (far * near)/(far - near)
            ndc[3,2] = 1.0

            proj[0,0] = alpha; proj[0,1] = skew ; proj[0,2] = u0 
            proj[1,1] = beta ; proj[1,2] = v0 
            proj[2,2] = 1; 
            proj[3,3] = 1.0
            return ndc, proj
        # def gen_proj(Q, left, right, bottom, top, near, far):
        #     near = near 
        #     far = far 
        #     Q = np.copy(Q)
        #     Q /= Q[-1,-1] # remove alpha

        #     alpha = Q[0,0]
        #     beta = Q[1,1]
        #     skew = Q[0,1]
        #     u0 = Q[0,2]
        #     v0 = Q[1,2]
            
        #     proj = np.zeros((4,4), dtype=np.float32)
        #     ndc = np.zeros((4,4), dtype=np.float32)

        #     ndc[0,0] = 2.0/(right - left) ; ndc[0,3] = -(right + left)/(right-left)
        #     ndc[1,1] = 2.0/(top - bottom) ; ndc[1,3] = -(top + bottom)/(top - bottom)
        #     ndc[2,2] = -2.0/(far - near) ; ndc[2,3] = -(far + near)/(far - near)
        #     ndc[3,3] = 1.0

        #     proj[0,0] = alpha; proj[0,1] = skew ; proj[0,2] = u0 
        #     proj[1,1] = beta ; proj[1,2] = v0 
        #     proj[2,2] = -(near+far); proj[2,3] = -near*far 
        #     proj[3,2] = 1.0
            
            return ndc, proj
        # ndc, new_proj = gen_proj(Q, 0, self.width, self.height, 0,0.01, 1000.0)
        # ndc, new_proj = gen_proj(Q, 0, self.width, self.height, 0,0.01, 1000.0)
        ndc, new_proj = gen_proj(Q, 0, self.width, 0, self.height,0.01, 1000.0)
        # ndc, new_proj = gen_proj(Q, -self.width/2, self.width/2, -self.height/2, self.height/2,0.01, 1000.0)
        # new_Q = new_NDC@new_Q
        xxx = np.identity(4, dtype=np.float32)
        xxx[2, 2] = -1
        # res3 = new_Rt@np.concatenate([v, np.ones((len(v),1))], axis=-1).T
        res11=new_Rt@np.concatenate([v, np.ones((len(v),1))], axis=-1).T
        # res11 = xxx @ res111
        te2 = Q@res11[:3, :]
        te23 =te2[:-1, :] / te2[-1, :]

        
        res32 =new_proj@res11
        # resr = res32 / res32[-2, :]
        res =ndc@res32
        res2 = res[:-1, :] / res[-1, :]
        rollback = self.width*(res2[1, :] + 1 )/2
        ww= res[0,:]*self.width
        ss = res[1,:]*self.height
        res4= np.array([[1,0,0],[0,-1,self.height],[0,0,1]], dtype=np.float32)@res2

        # new_Q = self.convert_hz_intrinsic_to_opengl_projection(Q,0,0, self.width, self.height, znear, zfar, "y down" )
        # new_Rt = xxx@new_Rt
        new_Rt = new_Rt
        new_proj = ndc@new_proj
        nn = new_proj@res11
        nn2 = nn/nn[-1,:]
   
        rollback2 = self.width*(nn2[1, :] + 1 )/2
        rollback3 = self.height*(nn2[2, :] + 1 )/2

        # trans = glm.mat4(1)
        # trans =glm.scale(trans, glm.vec3(0.1,0.1,0.1))
        # trans = glm.translate(trans, glm.vec3(0,0,-3))
        # camera = glm.lookAt(glm.vec3(0,0,5),glm.vec3(0,0,-1),glm.vec3(0,1,0))
        # # print(camera)
        # # camera = glm.translate(glm.vec3(0, 0, -10))
        # new_proj = glm.perspective(np.pi*0.5, self.width/self.height, 0.1, 1000)
        # new_Rt = glm.mul(camera, trans)


        Q_loc = glGetUniformLocation(self.gl_program, 'Q')
        Rt_loc = glGetUniformLocation(self.gl_program, 'Rt')
        light_pos_loc = glGetUniformLocation(self.gl_program, 'lightPos')
        light_color_loc = glGetUniformLocation(self.gl_program, 'lightColor')
        obj_color_loc = glGetUniformLocation(self.gl_program, 'objectColor')
        

        # glUniformMatrix4fv(Q_loc,1, GL_FALSE, glm.value_ptr(new_proj))
        # glUniformMatrix4fv(Rt_loc, 1, GL_FALSE, glm.value_ptr(new_Rt))

        glUniformMatrix4fv(Q_loc,1, GL_TRUE, new_proj)
        glUniformMatrix4fv(Rt_loc, 1, GL_TRUE, new_Rt)

        
        color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        light = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        mean_v = np.mean(v, axis=0).reshape(-1,1)
        lightPos = mean_v + np.array([[0],[0],[1000]], dtype=np.float32)
        # lightPos = np.array([[0],[0],[1000]], dtype=np.float32)

        # lightPos = np.array([0.0, 0.0, -0.0], dtype=np.float32)
        # lightPos = -new_Rt[:-1, -1].reshape(-1,).astype(np.float32)
        glUniform3f(obj_color_loc, *color.ravel())
        glUniform3f(light_pos_loc, *lightPos.ravel())
        glUniform3f(light_color_loc, *light.ravel())
        
        
        # # v_normal /= np.linalg.norm(v_normal, axis=-1).reshape(-1,1)
        # light_dir1 = lightPos.reshape(-1,3) - v.reshape(-1, 3)
        # light_dir = light_dir1/np.linalg.norm(light_dir1, axis=-1).reshape(-1,1)
        # diff1 = np.sum(light_dir*v_normal, -1)
        # diff = np.clip(diff1, a_min=0, a_max=1)
        # diffuse = diff.reshape(-1,1) * light
        # diffuse = diffuse*color

        # import matplotlib.pyplot as plt

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter([lightPos[0,0]],[lightPos[1,0]],[lightPos[2,0]], marker='^')
        # xx = v[:, 0]
        # yy = v[:, 1]
        # zz = v[:, 2]
        # ax.set_xlim(-30, 30)
        # ax.set_ylim(-30, 30)
        # ax.set_zlim(-30, 30)
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # dx = v_normal[:,0]
        # dy = v_normal[:,1]
        # dz = v_normal[:,2]
        # ax.scatter(xx, yy, zz, marker='o')
        # ax.quiver(xx,yy,zz, dx, dy, dz,color='red')
        # plt.show()

        glDrawElements(GL_TRIANGLES, f.size, GL_UNSIGNED_INT, ctypes.c_void_p(0))
        
        
        
        glUseProgram(0)
        res = self.read_color_buffer()
        # res = data =cv2.flip(res, 0)

        test = np.empty_like(flat_vnv, dtype=np.float32)
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, flat_vnv.itemsize*flat_vnv.size, test.ctypes.data_as(ctypes.c_void_p))

       
        mask = np.sum(res, axis= -1) != 0
        img[mask] = res[mask, :3]
        
        # return res[...,:3]
        return img

        
    

g_mesh_writer = MeshWriter()





def save(path, img):
    cv2.imwrite(path, img)



import igl 

def draw_mesh_to_img(img, Q, Rt, v, f, color, width = None ,caption = None ):
    """
    img : background image
    """
    img = np.copy(img)

    # v, f = igl.read_triangle_mesh("./prep_data/generic_neutral_mesh.obj")
    v = v.astype(np.float32)
    f = f.astype(np.uint32)
    vec1 = v[f[:, 1]] - v[f[:, 0]]
    vec2 = v[f[:, 2]] - v[f[:, 0]]
    v_normal = np.zeros_like(v)
    v_normal_denorm = np.zeros((len(v), 1))
    f_normal = np.cross(vec1, vec2, axis=-1)
    f_normal /= np.linalg.norm(f_normal, axis=-1).reshape(-1,1)
 
    for fi, (fn, (vi1,vi2,vi3)) in  enumerate(zip(f_normal, f)):
        v_normal[vi1] += fn
        v_normal[vi2] += fn
        v_normal[vi3] += fn
        v_normal_denorm[vi1] += 1
        v_normal_denorm[vi2] += 1
        v_normal_denorm[vi3] += 1
    v_normal /= v_normal_denorm
    v_normal = v_normal.astype(np.float32)
    v = v.astype(np.float32)

    img = g_mesh_writer.draw(img, Q, Rt, v, f, v_normal=v_normal, color = color)

    if width == None :
        resized_img = img
    else:
        resized_img = resize_img(img, width)
    if caption is not None:
        put_text(resized_img, caption, color, base_loc=(0,0), \
             font_face = 1, font_scale=2, thickness=1, linetype=cv2.LINE_AA, fit_bool=False )
    return resized_img




