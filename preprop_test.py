#
import cv2, glob, os 
import os.path as osp
import numpy as np 
import geo_func as geo 
import igl 
import scipy.optimize as opt
import re
import tqdm
import camera_clib as clib
lmk_idx = [
1278,
1272,
12,
1834,
243,
781,
2199,
1447,
966,
3661,
4390,
3022,
2484,
4036,
2253,
3490,
3496,
268,
493,
1914,
2044,
1401,
3615,
4240,
4114,
2734,
2509,
978,
4527,
4942,
4857,
1140,
2075,
1147,
4269,
3360,
1507,
1542,
1537,
1528,
1518,
1511,
3742,
3751,
3756,
3721,
3725,
3732,
5708,
5695,
2081,
0,
4275,
6200,
6213,
6346,
6461,
5518,
5957,
5841,
5702,
5711,
5533,
6216,
6207,
6470,
5517,
5966,
]
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



    def build(self, cutter = None):
        imgs, lmks_2d = self.load_data()
        meshes = self.load_mesh(cutter)

    def load_mesh(self, cutter = None):
        root_dir = self.mesh_dir

        identity_mesh_name = glob.glob(osp.join(root_dir, "**.obj"))
        identity_mesh_name.sort(key= natural_keys)
        identity_mesh_name = identity_mesh_name[:cutter]
        self.meshes = []
        for id_file_path in tqdm.tqdm(identity_mesh_name):
            mesh_collect = []
            id_name = osp.basename(id_file_path)
            
            v, f = igl.read_triangle_mesh(id_file_path)
            mesh_collect.append(v)
            expr_paths = glob.glob(osp.join(root_dir, "shapes", osp.splitext(id_name)[0], "**.obj"))
            expr_paths.sort(key= natural_keys)
            for expr_path in tqdm.tqdm(expr_paths, leave=False):
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

    def shape_fit(self, lmks_2d, meshes, lmk_idx):
        Q = clib.calibrate('./images/checker4/*.jpg')

        ids = np.array( [m[0] for m  in meshes])
        r,c = meshes[0][0].shape
        ids = ids.reshape(-1, 1, r, c)

        id_mean = np.mean(ids, axis=0, keepdims=True)
        id_mean_bar = id_mean[..., lmk_idx, :]
        ids_bar = ids[..., lmk_idx, :]
        _, _, new_r, new_c = ids_bar.shape
        exps = np.array([m[1:] for m  in meshes])
        lmks_2d = np.squeeze(np.array(lmks_2d))
        expr_mean = np.mean(exps, axis=0, keepdims=True)
        expr_mean_bar = exps[..., lmk_idx, :]
        expr_bar = exps[..., lmk_idx, :] - expr_mean_bar
        id_size, expr_size, _, _ =expr_bar.shape

        A = id_size*expr_size
        id_weight = np.zeros((id_size, 1)) # Identity
        exp_weight = np.zeros((id_size*expr_size, 1)) # Identity X Expression 

        ids_flatten = ids_bar.reshape(-1, new_r*new_c).T
        expr_flatten = expr_bar.reshape(-1, new_r*new_c).T

        def get_combine_bar_model(w_i, w_e):
            nonlocal id_mean_bar, expr_mean_bar, expr_bar, ids_bar
            return id_mean_bar + expr_mean_bar + expr_bar@w_e + ids_bar @ w_i
        def get_combine_model(w_i, w_e):
            nonlocal id_mean, expr_mean, exps, ids
            return id_mean + expr_mean + exps@w_e + ids @ w_i
        iter_num = 20
        def estimate_camera(lmk2d, vert_3d):
            ret, rvecs, tvecs = cv2.solvePnP(vert_3d, lmk2d)
            if ret:
                rot, _ =cv2.Rodrigues(rvecs[0])
                return rot, tvecs
            return None
        def transform_lm3d(v, cam_rot, cam_tvec):
            return v@cam_rot + cam_tvec
        def estimate_shape_coef():
            pass
        for i in tqdm(range(iter_num)):
            verts_3d = get_combine_model(id_weight, exp_weight)
            if i == 1:
                rot, tvecs = estimate_camera(lmks_2d, verts_3d) # camera matrix.(not homogenious)
            else:
                rot, tvecs = estimate_camera(lmks_2d, verts_3d) # camera matrix.(not homogenious)
            proj_all_lm3d = transform_lm3d()


        
if __name__ == "__main__":

    p = PreProp("lmks", "images", "prep_data")
    p.build(3)
    print(len(lmk_idx))
    # p.simple_camera_calibration(p.images[0], p.lmks[0], p.meshes[0][0], lmk_idx)
    p.shape_fit(p.lmks, p.meshes, lmk_idx)
