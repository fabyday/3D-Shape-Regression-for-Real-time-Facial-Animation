#
import cv2, glob, os 
import os.path as osp
import numpy as np 
import geo_func as geo 
import igl 
import scipy.optimize as opt
import re

def atof(text):
        try:
            retval = float(text)
        except ValueError:
            retval = text
        return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

class PreProp:
    def __init__(self, data_dir, img_dir, mesh_dir):
        
        self.proj = np.identity(4)
        self.rigid_trans = np.identity(4)
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.mesh_dir = mesh_dir



    def build(self):
        imgs, lmks_2d = self.load_data()
        meshes = self.load_mesh()

    def load_mesh(self):
        root_dir = self.mesh_dir

        identity_mesh_name = glob.glob(osp.join(root_dir, "**.obj"))
        identity_mesh_name.sort(key= natural_keys)
 
        self.meshes = []
        for id_file_path in identity_mesh_name:
            mesh_collect = []
            id_name = osp.basename(id_file_path)
            
            v, f = igl.read_triangle_mesh(id_file_path)
            mesh_collect.append(v)
            expr_paths = glob.glob(osp.join(root_dir, "shapes", id_name, "**.obj"))
            expr_paths.sort(key= natural_keys)
            for expr_path in expr_paths:
                v, f = igl.read_triangle_mesh(expr_path)
                mesh_collect.append(v)
            self.meshes.append(mesh_collect)

            



    def load_data(self):
        extension = [".jpeg", ".png", ".jpg"]
        lmk_data_files = glob.glob(osp.join(self.data_dir, "**.txt"))
        print("self.img_dir", self.img_dir)
        img_data_files = [name for ext in extension for name in glob.glob(osp.join(self.img_dir,"**"+ext)) ]
        self.images = []
        for fname in img_data_files:
            self.images.append(cv2.imread(fname))

        self.lmks = []
        for fname in lmk_data_files:
            lmk = []
            with open(fname, "r") as fp: 
                while True:
                    ss = fp.readline()
                    if not ss:
                        break
                    ss = ss.rstrip("\n")
                    x, y = ss.split(" ")
                    lmk.append([float(x),float(y)])
            self.lmks.append(lmk)
        return self.images, self.lmks 

    def fit_all_feature(self):
        pass
    def fit_identity(self):
        #fix all feature except identity
        pass

    def _shape_fit(self, lmks_2d, sel_ids, sel_exp):

        id_weight = np.zeros((len(sel_ids), 1)) # Identity
        exp_weight = np.zeros((len(sel_exp), 1)) # Identity X Expression

        num_ids = len(sel_exp)//len(sel_ids)

        A = sel_exp*num_ids
        def object_function(x, w_i, w_e, rot):
            # x 3xn or 4xn

  

            rotated_x = rot@x
            id_num = len(w_i)
            total_exp_elem_num = len(w_e)
            exp_num = total_exp_elem_num//id_num
            for i in range(id_num):
                w_e[i:(i+1)*exp_num, :] *= w_i(i)
            result = rotated_x @ w_e

        
        def jac(w):
            w_i = w[ : len(id_weight)]
            w_e = w[len(id_weight) : len(exp_weight)]
            rot = w[len(exp_weight) : ]
            delta = 0.0001
            result = np.zeros((3, len(w_i)+len(w_e)+len(rot)))
            for i in range(len(w_i)):
                w_i[i] += delta
                tmp = object_function(lmks_2d, w_i, w_e, rot)
                result[:, i] = tmp
                w_i[i] -= delta
            for i in range(len(w_e)):
                w_e[i] += delta
                tmp = object_function(lmks_2d, w_i, w_e, rot)
                result[:, len(w_i)+i] = tmp
                w_e[i] -= delta 
            for i in range(rot):
                rot[i] += delta
                tmp = object_function(lmks_2d, w_i, w_e, rot)
                result[:, len(w_i)+len(w_e)+i] = tmp
                rot[i] -= delta

        def wrap_obj(w):
            w_i = w[ : len(id_weight)]
            w_e = w[len(id_weight) : len(exp_weight)]
            rot = w[len(exp_weight) : ]
            lmks_2d
            return object_function(lmks_2d, w_i, w_e, rot)
        
        opt.minimize(fun = wrap_obj, x0 = 0 , method="L-BFGS-B", jac = jac)

        
        b  = lmks_2d.reshape(-1, 1)
        
        id_weight[0,0] = 1.0
        exp_weight[0,0] = 1.0
        self.a = 0.01


    def simple_camera_calibration(self, nuetral_img, lmks, neutral_mesh, lmk_index):
        # neutral_mesh vertice mapped to lmk

        h,w, _ = nuetral_img.shape
        u0 = w/2
        v0 = h/2

        A = np.zeros((68*2,9))

        for i, v, lmk in enumerate(zip(neutral_mesh, lmks)):
            A[2*i,:2] = v[:2]
            A[2*i, 2] = 1
            A[2*i, -3:-1] = -lmk[0]*v[:2]
            A[2*i, -1] = -lmk[0]*1
            
            A[2*i + 1,:2] = v[:2]
            A[2*i + 1, 2] = 1
            A[2*i + 1, -3:-1] = -lmk[1]*v[:2]
            A[2*i + 1, -1] = -lmk[1]*1

        s, v, vh = np.linalg.svd(A)
        a = vh[:, 0]
        H = a.reshape(3,3)
        

        def V(H, i, j):
            h1 = H[:,0]
            h2 = H[:,1]

            v12 = np.array([ [h1[1]*h2[1]] , 
                        [h1[1]*h2[2] + h1[2]*h2[1]] ,
                        [h1[3]*h2[1] + h1[1]*h2[3]], 
                        [h1[2]*h2[2]], 
                        [h1[3]*h2[2] + h1[2]*h2[3]], 
                        [h1[3]*h2[3]]
                        ])
        v12 = V(H, 1,2)
        v11_22 = V(H, 1,1) - V(H, 2,2)
        VV = np.vstack(v12, v11_22)
        s,v, vh = np.linalg.svd(VV)
        b = vh[:, 0]


        b_mat = np.array([[b[0], b[1], b[2]],
                            [b[1], b[3], b[4]],
                            [b[2], [4],b[5]]
                            ])
        
        K = np.linalg.cholesky(b_mat)



        proj = []
        cam_param = []


            


        
if __name__ == "__main__":

    p = PreProp("lmks", "images", "prep_data")
    p.build()
    p.simple_camera_calibration(p.images[0], p.lmks[0], p.meshes[0][0])
