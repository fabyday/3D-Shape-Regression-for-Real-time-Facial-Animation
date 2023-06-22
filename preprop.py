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
        ids_bar = ids[..., lmk_idx, :]
        _, _, new_r, new_c = ids_bar.shape
        exps = np.array([m[1:] for m  in meshes])
        expr_bar = exps[..., lmk_idx, :] - ids_bar
        id_size, expr_size, _, _ =expr_bar.shape
        lmks_2d = np.array(lmks_2d)

        A = id_size*expr_size
        id_weight = np.zeros((id_size, 1)) # Identity
        exp_weight = np.zeros((id_size*expr_size, 1)) # Identity X Expression 
        
        param_length = len(id_weight) + len(exp_weight) + 6 

        ids_flatten = ids_bar.reshape(-1, new_r*new_c).T
        expr_flatten = expr_bar.reshape(-1, new_r*new_c).T
        cam_params = np.zeros((3+3, 1)) # rot axis by 3, translation by 3
        def object_function(y, ws):
            w_i = ws[:id_size]
            w_e = ws[id_size:id_size+id_size*expr_size]
            cam_params = ws[id_size+id_size*expr_size:-1]
            scale = ws[-1]
            # cam_rotation = cam_params[:3, :]
            cam_translate = cam_params[3:, :]
            cam_mtx = geo.Euler_PYR(cam_params[0,0],cam_params[1,0],cam_params[2,0])[:3, :3]
            # x 3xn or 4xn
            # tmp= ids_flatten@w_i + expr_flatten@w_e

            size_t = len(w_e)//len(w_i)
            tmp = np.zeros((ids_flatten.shape[0],1))
            for i in range(len(w_i)):
                tmpo = expr_flatten[:, size_t*i:size_t*(i+1)]@w_e[size_t*i:size_t*(i+1),:]
                tmp += (tmpo+ids_flatten[:, i, None])*w_i[i, 0]

            # tmp= ids_flatten@ + expr_flatten@w_e
            # tmp  = tmp @ w_i
            tmp = tmp.reshape(-1 , 3)
            # tmp = tmp.reshape(-1 , 3)
            # tmp = cam_mtx @ tmp.T + cam_translate
            tmp = scale*(cam_mtx @ tmp.T + cam_translate)
            tmp = Q@tmp
            tmp = tmp.T
            tmp /= tmp[:, -1, np.newaxis]
            tmp = (np.squeeze(y) - tmp[:,:-1])
            result = np.sum(np.power(tmp, 2)) + (ws.T@ws)[0,0]
            return result 
        

        
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
        

        def coord_gradient(y, w, i, eps = 0.0001):
            a = object_function(y, w )
            w[i,0] += eps
            b = object_function(y, w )
            
            w[i,0] -= eps
            print((a-b), ": ", (a-b)/eps)
            return (a - b) / eps
            

        def coordinate_descent(obj, y, wi, we, cam_params, alpha =0.03):
            thetas = np.vstack([wi, we, cam_params, np.array([[1.0]])])
            print("start cost : ", object_function(y,thetas ))
            for i in range(20):
                for i in range(len(thetas)):
                    thetas[i] = thetas[i]  - alpha * coord_gradient(y,thetas, i )
                    print(object_function(y,thetas ))
            return thetas
        
        
        b  = lmks_2d.reshape(-1, 1)
        
        id_weight[0,0] = 1.0
        self.a = 0.01
        # res = opt.minimize(fun = wrap_obj, x0 = 0 , method="L-BFGS-B", jac = jac)
        res = coordinate_descent(object_function, lmks_2d, id_weight, exp_weight, cam_params)


    def simple_camera_calibration(self, nuetral_img, lmks, neutral_mesh, lmk_index):
        # neutral_mesh vertice mapped to lmk

        h,w, _ = nuetral_img.shape
        u0 = w/2
        v0 = h/2

        A = np.zeros((68*2,9))
        sel_neutral_mesh = neutral_mesh[lmk_index]

        for i, (v, lmk) in enumerate(zip(sel_neutral_mesh, lmks)):
            #ax 
            A[2*i,:2] = v[:2]
            A[2*i, 2] = 1
            A[2*i, -3:-1] = -lmk[0]*v[:2]
            A[2*i, -1] = -lmk[0]*1
            #ay
            A[2*i + 1,3:5] = v[:2]
            A[2*i + 1, 5] = 1
            A[2*i + 1, -3:-1] = -lmk[1]*v[:2]
            A[2*i + 1, -1] = -lmk[1]*1

        s, v, vh = np.linalg.svd(A)
        a = vh[:, -1]
        H = a.reshape(3,3)
        

        def V(H, i, j):
            h1 = H[:,0]
            h2 = H[:,1]

            v12 = np.array([ [h1[0]*h1[0]] , 
                        [h1[0]*h2[1] + h1[1]*h2[0]] ,
                        [h1[2]*h2[0] + h1[0]*h2[2]], 
                        [h1[1]*h2[1]], 
                        [h1[2]*h2[1] + h1[1]*h2[2]], 
                        [h1[2]*h2[2]]
                        ])
            return v12
        v12 = V(H, 1,2)
        v11_22 = V(H, 1,1) - V(H, 2,2)
        VV = np.vstack([v12.T, v11_22.T])
        s, v, vh = np.linalg.svd(VV)
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
    p.build(3)
    print(len(lmk_idx))
    # p.simple_camera_calibration(p.images[0], p.lmks[0], p.meshes[0][0], lmk_idx)
    p.shape_fit(p.lmks, p.meshes, lmk_idx)
