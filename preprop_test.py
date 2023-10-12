#
import cv2, glob, os 
import os.path as osp
import numpy as np 
import geo_func as geo 
import igl 
import scipy.optimize as opt
import re
import tqdm
import scipy.spatial as sp
import camera_clib as clib
import data_loader as dl
import open3d as o3d
import open3d.visualization.rendering as rendering

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
    def __init__(self, meta_dir, mesh_dir, lmk_meta = "./ict_lmk_info.yaml"):
        
        self.proj = np.identity(4)
        self.rigid_trans = np.identity(4)
        self.meta_location = meta_dir 
        self.lmk_meta_location = lmk_meta
        self.img_meta, self.img_root, self.img_file_ext,  = dl.load_extracted_lmk_meta(self.meta_location)
        self.full_index, self.eye, self.contour, self.mouse, self.eyebrow, self.nose  = dl.load_ict_landmark(lmk_meta)
        self.mesh_dir = mesh_dir



    def build(self, cutter = None):
        imgs = self.load_data()
        meshes = self.load_mesh(cutter)

    def load_mesh(self, cutter = None):
        root_dir = self.mesh_dir

        identity_mesh_name = glob.glob(osp.join(root_dir, "identity**.obj"))
        identity_mesh_name.sort(key= natural_keys)
        identity_mesh_name = identity_mesh_name[:cutter]
        self.meshes = []
        self.f = None 
        mesh_collect = []
        for id_file_path in tqdm.tqdm(identity_mesh_name):
            id_name = osp.basename(id_file_path)
            
            v, _ = igl.read_triangle_mesh(id_file_path)
            mesh_collect.append(v)
        
        self.neutral_mesh_v , self.neutral_mesh_f = igl.read_triangle_mesh(osp.join(root_dir, "generic_neutral_mesh.obj"))
        self.id_meshes = mesh_collect

        expr_paths = glob.glob(osp.join(root_dir, "shapes", "generic_neutral_mesh", "**.obj"))
        expr_paths.sort(key= natural_keys)
        mesh_collect = []
        for expr_path in tqdm.tqdm(expr_paths, leave=False):
            v, f = igl.read_triangle_mesh(expr_path)
            mesh_collect.append(v)
        self.meshes.append(mesh_collect)
        self.expr_meshes = mesh_collect

            



    def load_data(self):
        extension = [".jpeg", ".png", ".jpg"]

        def read_lmk_meta(path):
            lmk = []
            with open(path, "r") as fp: 
                while True:
                    ss = fp.readline()
                    if not ss:
                        break
                    ss = ss.rstrip("\n")
                    x, y = ss.split(" ")
                    lmk.append([float(x),float(y)])
            return lmk

        self.img_meta.keys()


        self.img_and_info = dict()
        self.img_list = []
        for category_idx, key in enumerate(self.img_meta.keys()):
            category = self.img_and_info.get(key, None)
            meta_item = self.img_meta[key] # into category
            if category == None :
                self.img_and_info[key] = []
                category = self.img_and_info[key]
            
            for meta_data in meta_item:
                meta_data['landmark']
                name = meta_data['name']
                lmk_data = read_lmk_meta(meta_data['landmark'])
                img_data = cv2.imread(osp.join(self.img_root, name+self.img_file_ext))
                img_data = {'category_index' : category_idx, 'index' : len(self.img_list), "name" : name, "lmk_data" : lmk_data, "img_data": img_data}
                self.img_list.append(img_data)
                category.append(img_data)

        return self.img_list
    


    def calc_cam_intrinsic_Q(self,guessed_focal_length, img_w, img_h):
        #naive

        h,w, = img_h, img_w
        uv = [w/2, h/2]
        Q = np.zeros((3,3))

        focal_length = guessed_focal_length
        Q[:2, -1] = uv
        Q[-1, -1] = 1
        Q[0, 0] = Q[1, 1] = focal_length
        
        return Q
    
    def projection(Q, V):
        new_v = (Q @ V.T).T
        new_v /= new_v[:, -1]
        return new_v[:, :-1]

    # def find_fit_projection_mat(self, Q, R, v, lmk):
    #     old_f = Q[0,0]
    #     u = Q[0, -1]
    #     v = Q[1, -1]
    #     rot_v_T = R@v.T
    #     rot_v_T[:-1, :] /= rot_v_T[-1, :]
    #     rot_v_T = rot_v_T[:-1, :]
        
    #     lmk[:, 0] -= u
    #     lmk[:, 1] -= v


    #     res = np.linalg.lstsq(rot_v_T.reshape(-1, 1), lmk.reshape(-1, 1))
    #     new_f = res[0]
    #     Q[0,0] = new_f
    #     Q[1,1] = new_f

    #     return Q        
    def find_fit_projection_mat(self, obj_func,neutral, ids, exprs, mesh_lmk_mapping_idx, id_weight, expr_weight, Q , R_t, lmk, max_focal_length = 100):
        """
        init_focal_length : pixel based focal length
        """
        
        old_f = Q[0,0]
        u = Q[0, -1]
        v = Q[1, -1]
        rot_v_T = R@v.T
        rot_v_T[:-1, :] /= rot_v_T[-1, :]
        rot_v_T = rot_v_T[:-1, :]
        
        lmk[:, 0] -= u
        lmk[:, 1] -= v
        


        fx  = 0
        fx_min = 0
        fx_max = max_focal_length
        def calc_Q(fx):

            reQ = np.copy(Q)
            reQ[0,0] = fx 
            reQ[1,1] = fx 
            return reQ
        def calc_func(Q_cur, Q_prev):
            f_current = obj_func(Q_cur, v, lmk)
            f_prev = obj_func(Q_prev, v, lmk)
            return f_current, f_prev
        k = fx_max - fx_min
        eps = 0.001
        fx = fx + k

        f_current, f_prev = calc_func(Q)
        while True(abs(f_current - f_prev) <= eps ):
            if f_prev >= f_current :
                
                f_current = calc_func(Q)


            
            f_current, f_prev = calc_func(Q)




        return Q        
        
    def convert_focal_length(w, h, mm_focal_length=36):
        """
        mm to pixel
        """
        m = max(h, w)
        standard_sensor_size = 36
        m *mm_focal_length / standard_sensor_size


    # def shape_fit(self, lmks_2d, images, id_meshes, expr_meshes, lmk_idx):
    def shape_fit(self, id_meshes, expr_meshes, lmk_idx):
        # Q = clib.calibrate('./images/checker4/*.jpg')
        # extract actor-specific blendshapes
        # iteratively fit shapes
        # it consists with 2 phases.
        # we find id weight, expr weight, proj matrix per each iteration, 
        # but we don't save and reuse each weights,
        # then why we should iterate optimization, 
        # The main reason why we calculate weights is find contour points and fit them to real face.
        # 
        
        lmk_idx = np.array(lmk_idx)
        lmk_idx_list = np.stack([lmk_idx for _ in range(len(self.img_list))],axis=0)

        neutral = self.neutral_mesh_v
        # neutral_bar = neutral[lmk_idx, :]

        ids = np.array(id_meshes)
        ids -= np.expand_dims(neutral, axis=0)
        # ids_bar = ids[..., lmk_idx, :]
        id_num,_,_ = ids.shape


        expr = np.array(expr_meshes)
        expr -= np.expand_dims(neutral, axis=0)
        # expr_bar = expr[..., lmk_idx, :]
        expr_num, _,_ = expr.shape
        
        # lmks_2d = np.array(lmks_2d)

        # def get_combine_bar_model( w_i, w_e):
            # nonlocal neutral_bar, expr_bar, ids_bar
        def get_combine_bar_model(neutral_bar,ids_bar, expr_bar, w_i, w_e):
            expr_num, expr_v_size, expr_dim = expr_bar.shape
            id_num, id_v_size, id_dim = ids_bar.shape 

            reshaped_expr = expr_bar.reshape(expr_num, expr_v_size*expr_dim).T
            reshaped_id = ids_bar.reshape(id_num, id_v_size*id_dim).T
            res = reshaped_id@w_i + reshaped_expr@w_e 
            # res = res.reshape(id_dim, id_v_size).T
            res = res.reshape(id_v_size, id_dim)
            return neutral_bar + res
        def get_combine_model(w_i, w_e):
            nonlocal neutral, expr, ids
            expr_num, expr_v_size, expr_dim = expr.shape
            new_exps = expr.reshape(expr_num, expr_v_size*expr_dim ).T
            id_num, id_v_size, id_dim = ids.shape 
            new_ids = ids.reshape(id_num, id_v_size*id_dim).T
            # exp_res =  new_exps@w_e
            # exp_res = exp_res.T.reshape(c,d)
            # id_res = new_ids@w_i
            # id_res = id_res.T.reshape(g,h)
            res = new_ids@w_i + new_exps@w_e
            res = res.reshape(id_v_size, id_dim)
            return neutral + res
        
        
        def estimate_camera(lmk2d, vert_3d, Q):
            
            rvec, tvec = cv2.solvePnPRefineLM(vert_3d, lmk2d, )
            rot, _ = cv2.Rodrigues(rvec)
            tvec = tvec.reshape(-1, 1)
            mat = np.concatenate([rot, tvec], axis = -1)

            # # lmk2d xyxy///
            # A = np.zeros((lmk2d.size, 8))
            # n, t = lmk2d.shape
            # A[0:2*n-1:2, 0:3] = vert_3d
            # A[0:2*n-1:2, 3] = 1
            # A[1:2*n:2, 4:-1] = vert_3d
            # A[1:2*n:2, 7] = 1
            # b = lmk2d.reshape(-1, 1)
            # res = np.linalg.lstsq(A, b)
            
            # rr = res[0]
            # r1 = rr[:3]
            # r2 = rr[4:-1]
            # sTx = rr[3]
            # sTy = rr[-1]

            # r1_norm = np.linalg.norm(r1)
            # r2_norm = np.linalg.norm(r2)
            # scale = (r1_norm + r2_norm)/2 
            # r1 = r1 / r1_norm
            # r2 = r2 / r2_norm
            # r3 = np.cross(r1.T, r2.T)
            # cam_rot = np.hstack([r1,r2,r3.T])


            # u,s,v = np.linalg.svd(cam_rot)
            # cam_rot = u@v
            
            # if np.linalg.det(np.diag(s)) < 0:
            #     u[-1, :] = -u[-1,:]
            #     cam_rot = u@v.T
            # cam_vec = np.array([sTx/scale, sTy/scale])
            # # test =scale*(cam_rot[:2, :] @ vert_3d.T + cam_vec)
            # # test_j = np.concatenate([cam_vec, np.array([[0]])], axis=0)
            # # test2 =scale*(cam_rot @ vert_3d.T + test_j)
            # # test2 /= test2[-1, :]
            # # test /= test[-1, :]
            # # print((test).T[:3])
            # # print(lmk2d[:3])
            # return scale, cam_rot, cam_vec
            return mat

        def transform_lm3d(v, scale, cam_rot, cam_tvec):
            return (scale*(cam_rot[:2, :]@v.T + cam_tvec[:2,:])).T
        
        # def estimate_shape_coef(scale, cam_rot, cam_tvecs, lmk_2d, reg_weight = 100): # phase 1 
            # nonlocal ids_bar, expr_bar, neutral_bar
        def estimate_shape_coef(ids_bar, expr_bar, neutral_bar, scale, cam_rot, cam_tvecs, lmk_2d, reg_weight = 100): # phase 1 
            neutral = neutral_bar
            id_num, id_v_size, id_dim = ids_bar.shape
            expr_num, expr_v_size, expr_dim = expr_bar.shape
            idexp_size = id_num+expr_num 
            
            datas = np.concatenate([ids_bar, expr_bar], axis =  0) # id+num + expr_num, v_size, 3
            datas = np.transpose(datas, [2, 1, 0])
            cam_rot = cam_rot[:2, :]
            dim, v_sizes, num = datas.shape
            datas = scale*cam_rot@datas.reshape(dim, v_sizes*num)
            datas = datas.reshape(2, v_sizes, num)
            datas = np.transpose(datas,[2,1,0])
            proj_neutral = cam_rot @ neutral.T

            
            A = np.zeros((lmk_2d.size + idexp_size, idexp_size))
            b = np.zeros((lmk_2d.size + idexp_size, 1))
            
            # main problem
            num, v_size, dim = datas.shape
            A[:lmk_2d.size, :] = datas.reshape(num, v_size*dim).T
            test = scale*(proj_neutral.T + cam_tvecs[:2].T)
            tmp_b = lmk_2d - scale*(proj_neutral.T + cam_tvecs[:2].T)
            tmp_b = tmp_b.reshape(-1, 1)
            b[:lmk_2d.size, :] = tmp_b
            for i in range(idexp_size):
                A[lmk_2d.size + i, i] = reg_weight
                b [lmk_2d.size + i, 0] = 0
            # add reg weight
            
            #solve

            coef = np.linalg.lstsq(A, b)
            id_coef = coef[0][ :id_num, :]
            expr_coef = coef[0][id_num:, :]
            # coef = opt.nnls(A,b.reshape(-1))
            # id_coef = coef[0][ :id_size]
            # expr_coef = coef[0][id_size:]
            return id_coef, expr_coef


        # def expr_weight_fit(scale, cam_rot, cam_tvecs, lmk_2d, reg_weight = 100):
            # nonlocal ids_bar, expr_bar, neutral_bar
        def expr_weight_fit(neutral_bar, ids_bar, expr_bar, scale, cam_rot, cam_tvecs, lmk_2d, id_weight, reg_weight = 100):
            neutral = neutral_bar
            expr_num, expr_v_size, expr_dim = expr_bar.shape
            expr_size = expr_num 
            
            datas = expr_bar
            datas = np.transpose(datas, [2, 1, 0])
            cam_rot = cam_rot[:2, :]
            dim, v_sizes, num = datas.shape
            datas = scale*cam_rot@datas.reshape(dim, v_sizes*num)
            datas = datas.reshape(2, v_sizes, num)
            datas = np.transpose(datas,[2,1,0])
            proj_neutral = cam_rot @ neutral.T

            
            A = np.zeros((lmk_2d.size + expr_size, expr_size))
            b = np.zeros((lmk_2d.size + expr_size, 1))
            
            id_num, id_v_size, id_dim = ids_bar.shape
            reshaped_id_bar = np.transpose(ids_bar, [2, 1, 0])
            reshaped_id_bar = reshaped_id_bar.reshape(id_dim, id_v_size*id_num)
            proj_id = cam_rot @ reshaped_id_bar
            exact_id = proj_id.reshape(-1, id_num) @ id_weight
            exact_id = exact_id.reshape(-1, id_v_size).T # v_size, 2


            # main problem
            num, v_size, dim = datas.shape
            A[:lmk_2d.size, :] = datas.reshape(num, v_size*dim).T
            test = scale*(proj_neutral.T + cam_tvecs[:2].T)
            # tmp_b = lmk_2d - scale*(proj_neutral.T + cam_tvecs[:2].T)
            tmp_b = lmk_2d - scale*(exact_id + proj_neutral.T + cam_tvecs[:2].T)
            tmp_b = tmp_b.reshape(-1, 1)
            b[:lmk_2d.size, :] = tmp_b
            for i in range(expr_size):
                A[lmk_2d.size + i, i] = reg_weight
                b [lmk_2d.size + i, 0] = 0
            # add reg weight
            
            #solve

            coef = np.linalg.lstsq(A, b)
            expr_coef = coef[0]
            # coef = opt.nnls(A,b.reshape(-1))
            # id_coef = coef[0][ :id_size]
            # expr_coef = coef[0][id_size:]
            return expr_coef
        def get_bars(neutral, ids, exprs, sel_lmk_idx):
            neutral_bar = neutral[sel_lmk_idx, :]
            ids_bar = ids[:, sel_lmk_idx, :]
            expr_bar = exprs[:, sel_lmk_idx, :]
            return neutral_bar, ids_bar, expr_bar
        

        
        def single_energy_term(neutral, ids, exprs, sel_lmk_idx, id_weight, expr_weight, Q, Rt, lmk):
            
            neutral_bar, ids_bar, expr_bar = get_bars(neutral, ids, exprs, sel_lmk_idx)
            R = Rt[:, :3]
            t = Rt[:, -1]
            Ax = (neutral_bar + ids_bar @ id_weight + expr_bar @ expr_weight)
            Ax = Q(R@Ax + t)
            Ax = Ax / Ax[ : , -1]
            Ax = Ax[:, :-1] # v, 2
            residual = Ax - lmk
            residual = residual.reshape(-1,1)
            m = residual.size
            result = 1/(2*m) * residual.T @ residual
            return result
        def multi_lmk_energy_term(neutral, ids, exprs, sel_lmk_idx, id_weight, expr_weights, Qs, Rts, lmks):
            residual = 0
            for expr_weight, Q, Rt, lmk in zip(expr_weights, Qs, Rts, lmks):
                residual += single_energy_term(neutral, ids, exprs, sel_lmk_idx, id_weight, expr_weight, Q, Rt, lmk)
            return residual
        def coordinate_descent(cost_function, init_weight, alpha, eps= 0.001, num_iters=4):
            n = init_weight.size
            new_weight1 = np.copy(init_weight)
            new_weight2 = np.zeros_like(new_weight1)

            for i in range(num_iters):
                for j in range(n):
                    #Coordinate descent in vectorized form
                    new_weight2[...] = new_weight1
                    new_weight2[j] = new_weight2[j] + eps
                    h = cost_function(new_weight1)
                    h_1 = cost_function(new_weight2)
                    gradient = h - h_1/eps
                    new_weight1[j] = new_weight1[j] - alpha * gradient
            return new_weight1

        def multi_lmk_expr_weight_fit(neutral, ids, exprs, sel_lmk_idx_list, lmks, Rt_list, Q_list, reg_weight = 100):
            # fitting multi image that share same expression.
            assert(len(sel_lmk_idx_list) == len(lmks))
        
            def get_bars(neutral, ids, exprs, sel_lmk_idx):
                neutral_bar = neutral[sel_lmk_idx, :]
                ids_bar = ids[:, sel_lmk_idx, :]
                expr_bar = exprs[:, sel_lmk_idx, :]
                return neutral_bar, ids_bar, expr_bar
            
            expr_num = exprs.shape[0]

            img_num = len(lmks)
            A = np.zeros((lmks[0].size*img_num + expr_num, expr_num))
            b = np.zeros((lmks[0].size*img_num + expr_num, 1))
            
            # main problem

            for i, (sel_lmk_idx, lmk, Rt, Q) in enumerate(zip(sel_lmk_idx_list, lmks, Rt_list, Q_list) ):
                neutral_bar, ids_bar, expr_bar = get_bars(neutral, ids, exprs, sel_lmk_idx)
                
                v_size, dim = neutral_bar.shape
                id_num, _, _ = ids_bar.shape
                expr_num , _, _ = expr_bar.shape

                expr_t = np.transpose(expr_bar, [2, 1, 0])
                expr_t = expr_t.reshape(dim, -1)
                id_t = np.transpose(ids_bar, [2,1,0])
                id_t = id_t.reshape(dim, -1)

                
                R = Rt[:, :3]
                t= Rt[-1, :]
                proj_neutral = R @ neutral_bar.T
                proj_neutral = proj_neutral.T # v_size, 3
                
                proj_expr = Q@ R @ expr_t
                # exact_expr = proj_expr.reshape(-1, expr_num) @ expr_weight
                # exact_expr = exact_expr.reshape(-1, v_size).T # v_size, 2
                proj_expr = proj_expr.reshape(2, v_size, expr_num)
                proj_expr = np.transpose(proj_expr,[2,1,0])
                proj_expr = proj_expr.reshape(expr_num, v_size*2).T

                proj_id = cam_rot @ id_t
                proj_id = proj_id.reshape(-1, id_num)
                proj_id = proj_id @ id_weight
                proj_id = proj_id.reshape(-1, v_size).T

                tmp_b = lmk - scale*(proj_id + proj_neutral + cam_tvec[:2].T)
                b[lmk.size*i:lmk.size*(i+1), :] += tmp_b.reshape(-1, 1)
                A[lmk.size*i:lmk.size*(i+1), :] += scale*proj_expr
            

            # num, v_size, dim = datas.shape
            # A[:lmk_2d.size, :] = datas.reshape(num, v_size*dim).T
            # test = scale*(proj_neutral.T + cam_tvecs[:2].T)
            # # tmp_b = lmk_2d - scale*(proj_neutral.T + cam_tvecs[:2].T)
            # tmp_b = lmk_2d - scale*(exact_id + proj_neutral.T + cam_tvecs[:2].T)
            # tmp_b = tmp_b.reshape(-1, 1)
            # b[:lmk_2d.size, :] = tmp_b
            
            # add reg weight
            for i in range(expr_num):
                A[lmk.size + i, i] = reg_weight
                b [lmk.size + i, 0] = 0
            #solve

            coef = np.linalg.lstsq(A, b)
            expr_coef = coef[0]
            residual = coef[1]
            # coef = opt.nnls(A,b.reshape(-1))
            # id_coef = coef[0][ :id_size]
            # expr_coef = coef[0][id_size:]
            return expr_coef, residual

        # def id_weight_fit(lmks, scales, cam_rots, cam_tvecs, expr_weights): # phase2
            # nonlocal ids_bar, expr_bar, neutral_bar # v_size, 3 
        def id_weight_fit(neutral, ids, exprs, sel_lmk_idx_list, lmks, scales, cam_rots, cam_tvecs, expr_weights): # phase2
            """
                lmks : all images lmk pts
            """
            # assert(lmk_size >= 1, "lmk size is not validate.")
            assert(len(expr_weights) >= 1)
            def get_bars(neutral, ids, exprs, sel_lmk_idx):
                neutral_bar = neutral[sel_lmk_idx, :]
                ids_bar = ids[:, sel_lmk_idx, :]
                expr_bar = exprs[:, sel_lmk_idx, :]
                return neutral_bar, ids_bar, expr_bar
            
            v_size = len(lmk_idx_list[0])
            _, dim = neutral.shape
            id_size, _, _ = ids.shape 
            expr_size, _, _ = exprs.shape

            lmk_size = np.array(lmks[0]['lmk_data']).size
            A = np.zeros((lmk_size*len(lmks) + id_size, id_size))
            b = np.zeros((lmk_size*len(lmks) + id_size, 1))
            # A = np.zeros((lmk_2d.size + id_size, id_size))
            # b = np.zeros_like(np.array(lmks[0]['lmk_data']).size + id_size)
            for i, (sel_lmk_idx, lmk, scale, cam_rot, cam_tvec, expr_weight) in enumerate(zip(sel_lmk_idx_list, lmks, scales, cam_rots, cam_tvecs, expr_weights) ):
                lmk = np.array(lmk['lmk_data']) 
                neutral_bar, ids_bar, expr_bar = get_bars(neutral, ids, exprs, sel_lmk_idx)
                    
                expr_t = np.transpose(expr_bar, [2, 1, 0])
                expr_t = expr_t.reshape(dim, -1)
                id_t = np.transpose(ids_bar, [2,1,0])
                id_t = id_t.reshape(dim, -1)



                cam_rot = cam_rot[:2, :]
                proj_neutral = cam_rot @ neutral_bar.T
                proj_neutral = proj_neutral.T # v_size, 3
                proj_expr = cam_rot @ expr_t
                exact_expr = proj_expr.reshape(-1, expr_size) @ expr_weight
                exact_expr = exact_expr.reshape(-1, v_size).T # v_size, 3
                proj_id = cam_rot @ id_t
                proj_id = proj_id.reshape(-1, id_size)
                tmp_b = lmk - scale*(exact_expr + proj_neutral + cam_tvec[:2].T)
                # b[:lmk_2d.size, :] += tmp_b.reshape(-1, 1)
                # A[:lmk_2d.size, :] += scale*proj_id
                b[lmk.size*i:lmk.size*(i+1), :] += tmp_b.reshape(-1, 1)
                A[lmk.size*i:lmk.size*(i+1), :] += scale*proj_id
            

            # regularization
            reg_weight = 100
            for i in range(id_size):
                A[lmk.size + i, i] = reg_weight
                b[lmk.size + i, 0] = 0
            res_id_weight = np.linalg.lstsq(A, b) 

            return res_id_weight[0]





        
        import copy 
        def draw_circle(v, img, colors = (1.0,0.0,0.0)):
            for vv in v:
                cv2.circle(img, center=vv.astype(int), radius=10, color=colors, thickness=2)

        def resize(img, width):
            h,w,c = img.shape
            ratio = width / w 
            new_h = h*ratio 
            img = cv2.resize(img, [int(new_h), int(width)])
            return img
        
        def calc_Rt():
            pass

        def find_contour(lmk2d, proj_3d_v):
            hull = sp.ConvexHull(proj_3d_v)
            convex_index = hull.vertices
            kd = sp.cKDTree(proj_3d_v[convex_index, :])
            d, idx = kd.query(lmk2d)
            return convex_index[idx]

        
        def draw_cv(index, expr_index, flag, id_weight, expr_weights, cam_scales, cam_rots, cam_tvecs):
            img = self.img_list[index]['img_data']
            truth = copy.deepcopy(img)
            test = copy.deepcopy(img)
            sel_lmk = np.array(self.img_list[index]['lmk_data'])
            exp_weight = expr_weights[expr_index]
            
            sel_index_list = lmk_idx_list[index]
            verts_3d = get_combine_bar_model(neutral[sel_index_list, :], ids[:,sel_index_list,:], expr[:,sel_index_list,:], id_weight, exp_weight)
            out_of_concern_idx = [i for i in range(len(neutral)) if i not in sel_index_list]
            non_sel_mesh_v = get_combine_model(id_weight, exp_weight)
            non_sel_mesh_v = non_sel_mesh_v[out_of_concern_idx]
            # draw_circle(transform_lm3d(verts_3d, 1, np.eye(3,3), np.zeros((3,1))), test, (255,0,0))
            non_sel_mesh_v = non_sel_mesh_v[::int(len(non_sel_mesh_v)//100),:]
            draw_circle(transform_lm3d(non_sel_mesh_v, cam_scales[index],cam_rots[index],cam_tvecs[index]), test, (0,255,0))
            draw_circle(transform_lm3d(verts_3d, cam_scales[index],cam_rots[index],cam_tvecs[index]), test, (0,0,255))
            draw_circle(sel_lmk, truth, (255,0,0))
            truth = resize(truth, 800)
            test = resize(test, 800)
            show_img = np.concatenate([truth, test], 1)
            cv2.imshow("test", show_img)
            
            path_name = osp.join("testdir", str(index))
            if not os.path.exists(path_name):
                os.makedirs(path_name)
            vv = get_combine_model(id_weight, exp_weight)
            igl.write_triangle_mesh(os.path.join(path_name, "test" + ".obj"), vv, self.neutral_mesh_f)
            if not flag:
                key = cv2.waitKey(0)
            else:
                key = cv2.waitKey(100)

            if key == ord('q'):
                return False
            elif key == ord('a'):
                return True
            else :
                return True
        


        def draw_contour(img, lmk, new_contour, orig, flag):
            draw_circle(lmk,img, colors=(255,0,0))
            draw_circle(new_contour,img, colors=(0,0,255))
            draw_circle(orig, img, colors=(0,255,255))
            for i in range(len(new_contour) -1):
                cv2.line(img, new_contour[i].astype(int), new_contour[i+1].astype(int), color=(0,255,0), thickness=3)

            for i in range(len(orig) -1):
                cv2.line(img, orig[i].astype(int), orig[i+1].astype(int), color=(255,255,0), thickness=3)
            img= resize(img, 1500)
            cv2.imshow("contour", img)
            if not flag:
                key = cv2.waitKey(0)
            else:
                key = cv2.waitKey(100)

            if key == ord('q'):
                return False
            elif key == ord('a'):
                return True
            else :
                return True



        def id_weight_fit_builder(sel_lmk_idx, expr_weights, Qs, Rts, lmks):
            def wrapper(id_weight):
                multi_lmk_energy_term(neutral, ids, expr, sel_lmk_idx, id_weight, expr_weights, Qs, Rts, lmks )

        def expr_weight_fit_builder(sel_lmk_idx, id_weight, Q, Rt, lmk ):
            def wrapper(expr_weight):
                single_energy_term(neutral, ids, expr, sel_lmk_idx, id_weight, expr_weight, Q, Rt, lmk )
        
        iter_num = 10
        expr_weights = [np.zeros((expr_num, 1)) for _ in range(len(self.img_and_info.keys()))]
        id_weight = np.zeros((id_num, 1))
        time_t = True
        Q_list = [[] for _ in range(len(self.img_list))]
        Rt_list = [[] for _ in range(len(self.img_list))]
        for i in tqdm.tqdm(range(iter_num)):
            
            for key_id, item in tqdm.tqdm(enumerate(self.img_and_info.values())):
                sel_imgs = [info['img_data'] for info in item ]
                lmk_2ds = np.array([info['lmk_data'] for info in item ])
                
                expr_Q_list = [ self.calc_cam_intrinsic_Q(0, info['img_data'].shape[1], info['img_data'].shape[0]) for info in item ]
                expr_Rt_list = [calc_Rt() for info in item ] # TODO 


                index_list =  [ info['index'] for info in item ]
                sel_lmk_idx_list = [ lmk_idx_list[info['index']] for info in item ]
                verts_3ds = [get_combine_bar_model( neutral[sel_lmk_idx, :], ids[:, sel_lmk_idx,:], expr[:, sel_lmk_idx, :], id_weight, expr_weights[key_id]) for sel_lmk_idx in sel_lmk_idx_list]
                

                for ii, Q, Rt in zip(index_list, expr_Q_list,expr_Rt_list):
                    Q_list[ii] = Q
                    Rt_list[ii] = Rt
                
                exp_weight = coordinate_descent(neutral, ids, expr, sel_lmk_idx_list ,lmk_2ds, expr_Rt_list, expr_Q_list)
                for local_i, ii, Q, Rt, mesh_lmk_mapping_idx in enumerate(zip(self, index_list, Q_list, Rt_list, sel_lmk_idx_list)):
                    h,w, _ = sel_imgs[local_i].shape
                    new_Q = self.find_fit_projection_mat(single_energy_term, neutral, ids, expr, mesh_lmk_mapping_idx, id_weight, exp_weight, Q, Rt, self.convert_focal_length(h,w, 50))
                    Q_list[ii] = new_Q 
                expr_weights[key_id] = exp_weight
            #===========================================================================================
                # expr_weights.append(exp_weight)
                for index in index_list:
                    # time_t = draw_cv(index, time_t, id_weight, expr_weights, cam_scales, cam_rots, cam_tvecs)
                    time_t = draw_cv(index,key_id, time_t, id_weight, expr_weights, cam_scales, cam_rots, cam_tvecs)
                    path_name = osp.join("testdir", str(index))
                    if not os.path.exists(path_name):
                        os.makedirs(path_name)
                    vv = get_combine_model(id_weight, exp_weight)
                    igl.write_triangle_mesh(os.path.join(path_name, "opt1_iter_num_" +str(i)+ ".obj"), vv, self.neutral_mesh_f)
            
        
            # phase 2
            # calculate exact id_weights and contour(optional)
            # res_id_weight = id_weight_fit(self.img_list, cam_scales, cam_rots, cam_tvecs, expr_weights)
            res_id_weight = id_weight_fit(neutral, ids, expr,lmk_idx_list ,self.img_list, cam_scales, cam_rots, cam_tvecs, expr_weights)

            # time_t = draw_cv(index, time_t, id_weight, expr_weights, cam_scales, cam_rots, cam_tvecs)

            id_weight = res_id_weight
            tmp_time_t = time_t
            time_t = True
            # contour
            # for index in range(len(self.img_list)):
            #     verts_3d = get_combine_model(id_weight, exp_weight)
            #     proj_3d_v = transform_lm3d(verts_3d, cam_scales[index],cam_rots[index],cam_tvecs[index])
            #     sel_lmk_2d = np.array(self.img_list[index]['lmk_data'])
            #     contour_lmk = sel_lmk_2d[ self.contour['full_index'] ]
            #     new_contour_idx = find_contour(lmk2d = contour_lmk, proj_3d_v = proj_3d_v)
            #     original_contour = lmk_idx_list[index][self.contour['full_index']]
            #     lmk_idx_list[index][self.contour['full_index']] = new_contour_idx

            #     time_t =  draw_contour(copy.deepcopy(self.img_list[index]['img_data']), sel_lmk_2d, proj_3d_v[new_contour_idx], proj_3d_v[original_contour], time_t)
            for img_idx in range(len(self.img_list)):
                exp_weight = expr_weights[self.img_list[img_idx]['category_index']]
                index = self.img_list[img_idx]['index']
                verts_3d = get_combine_model(id_weight, exp_weight)
                proj_3d_v = transform_lm3d(verts_3d, cam_scales[index],cam_rots[index],cam_tvecs[index])
                sel_lmk_2d = np.array(self.img_list[img_idx]['lmk_data'])
                contour_lmk = sel_lmk_2d[ self.contour['full_index'] ]
                new_contour_idx = find_contour(lmk2d = contour_lmk, proj_3d_v = proj_3d_v)
                original_contour = lmk_idx_list[img_idx][self.contour['full_index']]
                lmk_idx_list[img_idx][self.contour['full_index']] = new_contour_idx

                time_t =  draw_contour(copy.deepcopy(self.img_list[img_idx]['img_data']), sel_lmk_2d, proj_3d_v[new_contour_idx], proj_3d_v[original_contour], time_t)
            
            time_t = tmp_time_t            
            path_name = osp.join("testdir", "iden")
            if not os.path.exists(path_name):
                os.makedirs(path_name)
            vv = get_combine_model(id_weight, np.zeros_like(exp_weight))
            igl.write_triangle_mesh(os.path.join(path_name, "identity_iter_num_"+str(i) + ".obj"), vv, self.neutral_mesh_f)

            for key_id, item in enumerate(self.img_and_info.values()):
                for it in item:
                    index = it['index']
                    img = copy.deepcopy(self.img_list[index]['img_data'])
                    exp_weight = expr_weights[key_id]
                    draw_circle(np.array(self.img_list[index]['lmk_data']), img, (0,0,255))
                    img = resize(img, 800)
                    path_name = osp.join("testdir", str(index))
                    if not os.path.exists(path_name):
                        os.makedirs(path_name)
                    vv = get_combine_model(id_weight, exp_weight)
                    cv2.imwrite(osp.join(path_name, "img.jpeg"), img)
                    igl.write_triangle_mesh(os.path.join(path_name, "opt2_iter_"+str(i) + ".obj"), vv, self.neutral_mesh_f)
                
    def extract_train_set_blendshapes(train_set_images_lmks, neutral_pose, blendshapes):
        """
            this method use shape_fit method's actor(user)-specific blendshapes result.
            neutral pose : user-specific neutral pose
            blendshapes : user specific blendshapes
            ===========================================================================
            return
                weights that are extracted from user-specific blendshapes.
        """
        v_size, dim = neutral_pose.shape
        b_size, _, _ = blendshapes


        

        


        
if __name__ == "__main__":
    p = PreProp("landmark/meta.yaml", "prep_data")
    p.build()
    print(len(lmk_idx))
    # p.simple_camera_calibration(p.images[0], p.lmks[0], p.meshes[0][0], lmk_idx)
    p.shape_fit(p.id_meshes, p.expr_meshes, lmk_idx)

