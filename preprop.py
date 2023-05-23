#
import cv2, glob, os 
import os.path as osp
import numpy as np 
import geo_func as geo 



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
        


        
        b  = lmks_2d.reshape(-1, 1)
        
        id_weight[0,0] = 1.0
        exp_weight[0,0] = 1.0
        self.a = 0.01





        
        self.build