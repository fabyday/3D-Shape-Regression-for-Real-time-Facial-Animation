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
        self.f = None 
        for id_file_path in tqdm.tqdm(identity_mesh_name):
            mesh_collect = []
            id_name = osp.basename(id_file_path)
            
            v, f = igl.read_triangle_mesh(id_file_path)
            self.f = f
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

    def shape_fit(self, lmks_2d, images, meshes, lmk_idx):
        # Q = clib.calibrate('./images/checker4/*.jpg')

        ids = np.array( [m[0] for m  in meshes])
        r,c = meshes[0][0].shape
        ids = ids.reshape(-1, 1, r, c)

        id_mean = np.mean(ids, axis=0, keepdims=True)
        id_mean_bar = id_mean[..., lmk_idx, :]
        ids_bar = ids[..., lmk_idx, :]
        _, _, new_r, new_c = ids_bar.shape
        exps = np.array([m[1:] for m  in meshes])
        lmks_2d = np.squeeze(np.array(lmks_2d))
        a,b,c,d=exps.shape
        expr_mean = np.mean(exps.reshape(-1,c,d), axis=0, keepdims=True)
        expr_mean_bar = expr_mean[..., lmk_idx, :]
        expr_bar = exps[..., lmk_idx, :] - expr_mean_bar
        id_size, expr_size, _, _ =expr_bar.shape

        A = id_size*expr_size
        id_weight = np.zeros((id_size, 1)) # Identity
        exp_weight = np.zeros((id_size*expr_size, 1)) # Identity X Expression 

        ids_flatten = ids_bar.reshape(-1, new_r*new_c).T
        expr_flatten = expr_bar.reshape(-1, new_r*new_c).T

        def get_combine_bar_model(w_i, w_e):
            nonlocal id_mean_bar, expr_mean_bar, expr_bar, ids_bar
            a,b,c,d = expr_bar.shape
            expr_bar_tmp = np.reshape(expr_bar, (a*b, c*d)).T
            expr_bar_tmp = expr_bar_tmp@w_e 
            expr_bar_tmp = expr_bar_tmp.T.reshape(c, d)
            a,b,c,d = ids_bar.shape
            id_bar_tmp = np.reshape(ids_bar,(a*b, c*d)).T
            id_bar_tmp = id_bar_tmp @ w_i
            id_bar_tmp = id_bar_tmp.T.reshape(c,d)
            return np.squeeze(id_mean_bar) + np.squeeze(expr_mean_bar) + expr_bar_tmp + id_bar_tmp
        def get_combine_model(w_i, w_e):
            nonlocal id_mean, expr_mean, exps, ids
            a,b,c,d = exps.shape
            new_exps = exps.reshape(a*b, c*d ).T
            e,f,g,h = ids.shape 
            new_ids = ids.reshape(e*f, g*h).T
            exp_res =  new_exps@w_e
            exp_res = exp_res.T.reshape(c,d)
            id_res = new_ids@w_i
            id_res = id_res.T.reshape(g,h)
            
            return np.squeeze(id_mean) + np.squeeze(expr_mean) + id_res + exp_res
        iter_num = 20
        def estimate_camera(lmk2d, vert_3d, image):
            # ret, rvecs, tvecs = cv2.solvePnP(vert_3d, lmk2d)
            a,b,c = image.shape
            objp = [np.zeros((6*7,3), np.float32)]
            # q  = [np.zeros((6*7,2), np.float32)]
            # ret, cam_mat, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(objp, q, (b, a), None , None)
            # init_cam_mat = cv2.initCameraMatrix2D([vert_3d.astype(np.float32)], [lmk2d.astype(np.float32)], (b,c))
            # ret, cam_mat, dist_coeff, rvecs, tvecs = cv2.calibrateCamera([vert_3d.astype(np.float32)], [lmk2d.astype(np.float32)], (b, a), init_cam_mat, np.array([0,0,0,0.], np.float32), flags=cv2.CALIB_USE_INTRINSIC_GUESS)
            # if ret:
            #     rot, _ =cv2.Rodrigues(rvecs[0])
            #     # return rot, tvecs
            # cam_rot = cam_mat@rot
            # cam_vec = cam_mat@tvecs[0]
            # # cam_rot = rot
            # # cam_vec = tvecs[0]
            # test2 = cam_rot[:, 0].T @cam_rot[:, 1]/(np.linalg.norm(cam_rot[:, 1])*np.linalg.norm(cam_rot[:, 0]))
            # print(test2)
            # # r1_norm = np.linalg.norm(cam_rot[:, 0])
            # # r2_norm = np.linalg.norm(cam_rot[:, 1])
            # # cam_rot[:, 0] /= r1_norm
            # # cam_rot[:, 2] /= r2_norm
            # # r3_norm = np.linalg.norm(cam_rot[:, -1] )
            # # cam_rot /= r3_norm.reshape(-1,1)
            # r1_norm = np.linalg.norm(cam_rot[:, 0])
            # r2_norm = np.linalg.norm(cam_rot[:, 1])
            # scale = (r1_norm + r2_norm)/2
            # cam_rot[:, 0] = cam_rot[:, 0]/ r1_norm
            # cam_rot[:, 1] = cam_rot[:, 1]/ r2_norm
            # cam_rot[:, 2] = np.cross(cam_rot[:, 0], cam_rot[:, 1])
            
            # lmk2d xyxy///
            A = np.zeros((lmk2d.size, 8))
            n, t = lmk2d.shape
            A[0:2*n-1:2, 0:3] = vert_3d
            A[0:2*n-1:2, 3] = 1
            A[1:2*n:2, 4:-1] = vert_3d
            A[1:2*n:2, 7] = 1
            b = lmk2d.reshape(-1, 1)
            res = np.linalg.lstsq(A, b)

            rr = res[0]
            r1 = rr[:3]
            r2 = rr[4:-1]
            sTx = rr[3]
            sTy = rr[-1]

            r1_norm = np.linalg.norm(r1)
            r2_norm = np.linalg.norm(r2)
            scale = (r1_norm + r2_norm)/2 
            r1 = r1/r1_norm
            r2 = r2/r2_norm
            r3 = np.cross(r1.T, r2.T)
            cam_rot = np.hstack([r1,r2,r3.T])


            u,s,v = np.linalg.svd(cam_rot)
            cam_rot = u@v
            
            if np.linalg.det(np.diag(s)) < 0:
                u[-1, :] = -u[-1,:]
                cam_rot = u@v.T
            cam_vec = np.array([sTx/scale, sTy/scale])
            # test =scale*(cam_rot[:2, :] @ vert_3d.T + cam_vec)
            # test_j = np.concatenate([cam_vec, np.array([[0]])], axis=0)
            # test2 =scale*(cam_rot @ vert_3d.T + test_j)
            # test2 /= test2[-1, :]
            # test /= test[-1, :]
            # print((test).T[:3])
            # print(lmk2d[:3])
            return scale, cam_rot, cam_vec
        def transform_lm3d(v, scale, cam_rot, cam_tvec):
            return (scale*(cam_rot[:2, :]@v.T + cam_tvec[:2,:])).T
        def estimate_shape_coef(scale, cam_rot, cam_tvecs, lmk_2d, reg_weight = 5):
            nonlocal ids_bar, expr_bar, id_mean_bar, expr_mean_bar
            data_mean = id_mean_bar + expr_mean_bar
            id_size, id_row_size, id_col = np.squeeze(ids_bar).shape
            id_expr_size, expr_size, expr_row_size, ex_col = expr_bar.shape
            idexp_size = id_size+id_expr_size*expr_size 
            
            datas = np.concatenate([np.squeeze(ids_bar), expr_bar.reshape(-1, expr_row_size, ex_col)], axis =  0) # id_row*expr_row, v_size, 3
            datas = np.transpose(datas, [2, 1, 0])
            cam_rot = cam_rot[:2, :]
            a,b,c = datas.shape
            datas = scale*cam_rot@datas.reshape(a, b*c)
            datas = datas.reshape(2, b, c)
            datas = np.transpose(datas,[2,1,0])
            data_mean2 = cam_rot @ np.squeeze(data_mean).T

            
            A = np.zeros((lmk_2d.size + idexp_size, idexp_size))
            b = np.zeros((lmk_2d.size + idexp_size, 1))
            
            # main problem
            d,e,f = datas.shape
            A[:lmk_2d.size, :] = datas.reshape(d, e*f).T

            tmp_b = lmk_2d - scale*(data_mean2.T + cam_tvecs[:2].T)
            tmp_b = tmp_b.reshape(-1, 1)
            b[:lmk_2d.size, :] = tmp_b
            for i in range(idexp_size):
                A[lmk_2d.size + i, :i] = reg_weight
                b [lmk_2d.size + i, 0] = 0
            # add reg weight
            
            #solve
            coef = np.linalg.lstsq(A, b)
            id_coef = coef[0][ :id_size, :]
            expr_coef = coef[0][id_size:, :]
            return id_coef, expr_coef
        
        import copy 
        img = copy.deepcopy(images[0])
        def draw_circle(v, img):
            for vv in v:
                cv2.circle(img, center=vv.astype(int), radius=10, color=(1,0,0), thickness=2)

        def resize(img, width):
            h,w,c = img.shape
            ratio = width / w 
            new_h = h*ratio 
            img = cv2.resize(img, [int(new_h), int(width)])
            return img


        verts_3d = get_combine_bar_model(id_weight, exp_weight)    
        draw_circle(transform_lm3d(verts_3d, 1, np.eye(3,3), np.zeros((3,1))), img)

        img = resize(img, 800)
        cv2.imshow("test", img)
        cv2.waitKey(1000)
        
        for i in tqdm.tqdm(range(iter_num)):
            verts_3d = get_combine_bar_model(id_weight, exp_weight)
            if i == 0:
                scale, rot, tvecs = estimate_camera(lmks_2d, verts_3d, images[0]) # camera matrix.(not homogenious)
            else:
                scale, rot, tvecs = estimate_camera(lmks_2d, verts_3d, images[0]) # camera matrix.(not homogenious)
            img = copy.deepcopy(images[0])
            draw_circle(transform_lm3d(verts_3d,scale, rot, tvecs), img)
            img = resize(img, 800)
            cv2.imshow("test", img)
            cv2.waitKey(1000)
            # proj_all_lm3d = transform_lm3d(verts_3d,rot, tvecs)
            id_weight, exp_weight = estimate_shape_coef(scale, rot, tvecs, lmks_2d)
            path_name = "testdir"

            vv = get_combine_model(np.zeros_like(id_weight), np.zeros_like(exp_weight))
            igl.write_triangle_mesh(os.path.join(path_name, "init" + ".obj"), vv, self.f)
            vv = get_combine_model(id_weight, exp_weight)
            if not os.path.exists(path_name):
                os.makedirs(path_name)
            igl.write_triangle_mesh(os.path.join(path_name, str(i) + ".obj"), vv, self.f)
            with open(osp.join(path_name, str(i)+".txt"), "w") as fp:
                for iii, idw in enumerate(id_weight.reshape(-1)):
                    fp.write("id_weight : " + str(idw)+"\n")
                    fp.write("expr_weight : \n")
                    np.savetxt(fp,exp_weight.reshape(id_weight.size, -1)[iii].reshape(1,-1), fmt = "%3.3f")



        
if __name__ == "__main__":

    p = PreProp("lmks", "images", "prep_data")
    p.build(3)
    print(len(lmk_idx))
    # p.simple_camera_calibration(p.images[0], p.lmks[0], p.meshes[0][0], lmk_idx)
    p.shape_fit(p.lmks,p.images, p.meshes, lmk_idx)
