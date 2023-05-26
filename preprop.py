#
import cv2, glob, os 
import os.path as osp
import numpy as np 
import geo_func as geo 

import scipy.optimize as opt

class PreProp:
    def __init__(self, data_dir):
        
        self.proj = np.identity(4)
        self.rigid_trans = np.identity(4)
        self.data_dir = data_dir



    def build(self, id_meshes, exp_meshes, mesh_lmk_indices):
        imgs, lmks_2d = self.load_data()

    def load_data(self):
        extension = [".jpeg", ".png", ".jpg"]
        lmk_data_files = glob.glob(osp.join(self.data_dir, "/**.txt"))
        img_data_files = [glob.glob(osp.join(self.data_dir, ext)) for ext in extension]

        self.images = []
        for fname in img_data_files:
            self.images.append(cv2.imread(fname))

        for fname in lmk_data_files:
            with open(fname, "r") as fp : 
                ss = fp.readline()
                ss.split(" ")


    def fit_all_feature(self):
        pass
    def fit_identity(self):
        #fix all feature except identity
        pass

    def _shape_fit(self, lmks_2d, sel_ids, sel_exp):
        id_weight = np.zeros((len(sel_ids), 1)) # Identity
        exp_weight = np.zeros((len(sel_exp), 1)) # Identity X Expression

        def loss(x, A, b):
            crd_3d = A@x
            crd_2d = self.rigid_trans @ crd_3d
            return np.linalg.norm(crd_2d - b)

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

        lmks_2d
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





        
        self.build