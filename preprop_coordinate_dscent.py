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

import scipy 

import scipy.optimize as opt

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
        pass
        
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
            res = reshaped_id@w_i.reshape(-1,1) + reshaped_expr@w_e.reshape(-1,1)
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
        
        def get_bars(neutral, ids, exprs, sel_lmk_idx):
            neutral_bar = neutral[sel_lmk_idx, :]
            ids_bar = ids[:, sel_lmk_idx, :]
            expr_bar = exprs[:, sel_lmk_idx, :]
            return neutral_bar, ids_bar, expr_bar
                    
        
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
        

        def find_contour(lmk2d, proj_3d_v):
            hull = sp.ConvexHull(proj_3d_v)
            convex_index = hull.vertices
            kd = sp.cKDTree(proj_3d_v[convex_index, :])
            d, idx = kd.query(lmk2d)
            return convex_index[idx]

        
        def draw_cv(index, expr_index, flag, id_weight, expr_weights, Q_list, Rt_list):
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
            draw_circle(add_Rt_to_pts(Q_list[index], Rt_list[index], verts_3d), test, (0,0,255))
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
        
        
                
        def get_Rt(theta_x, theta_y, theta_z, tx, ty, tz):
            Rx = np.eye(3,3)
            Ry = np.eye(3,3)
            Rz = np.eye(3,3)

            Rx[1,1] = np.cos(theta_x); Rx[1,2] = -np.sin(theta_x)
            Rx[2,1] = np.sin(theta_x); Rx[2,2] = np.cos(theta_x)

            Ry[0,0] = np.cos(theta_y); Ry[0,2] = np.sin(theta_y)
            Ry[2,0] = -np.sin(theta_y); Ry[2,2] = np.cos(theta_y)
            
            Rz[0,0] = np.cos(theta_z); Rz[0,1] = -np.sin(theta_z)
            Rz[1,0] = np.sin(theta_z); Rz[1,1] = np.cos(theta_z)

            res = np.zeros((3,4))

            res[:, -1] = np.array([tx, ty, tz])
            res[:3, :3] =Rz@Ry@Rx

            return res
        

        def decompose_Rt(Rt):
            """
            input : 
                Z,y,x sequentialy
            return 
                x, y, z angle
            see also https://en.wikipedia.org/wiki/Rotation_matrix
            """
            y_angle = np.arctan2(-1*Rt[2,0],np.sqrt(Rt[0,0]**2+Rt[1,0]**2))
            x_angle = np.arctan2(Rt[2,1]/np.cos(y_angle),Rt[2,2]/np.cos(y_angle))
            z_angle = np.arctan2(Rt[1,0]/np.cos(y_angle),Rt[0,0]/np.cos(y_angle))


            return x_angle, y_angle, z_angle
            
        def add_Rt_to_pts(Q, Rt, x):
            R = Rt[:3,:3]
            t = Rt[:, -1, None]
            xt = x.T
            Rx = R @ xt 
            Rxt = Rx+t
            pj_Rxt = Q @ Rxt
            res = pj_Rxt/pj_Rxt[-1, :]
            return res[:2, :].T
            

        def id_weight_function(fixed_Q, fixed_Rt, nuetral_bar, ids_bar, exprs_bar, expr_weight_list, x, y):
            
            total_z = 0
            for w_e in expr_weight_list:
                verts3d = get_combine_bar_model(nuetral_bar, ids_bar, exprs_bar, x, w_e)
                vers_2d = add_Rt_to_pts(fixed_Q, fixed_Rt, verts3d)  
                z = vers_2d - y              
                z = z.reshape(-1,1)
                z = z.T@z 
                total_z += z 
            return total_z 

        def default_cost_function(Q, Rt, neutral_bar, ids_bar, exprs_bar, id_weight, expr_weight, y):
            # x := Q + Rt + expr weight

            blended_pose = get_combine_bar_model(neutral_bar , ids_bar, exprs_bar, id_weight, expr_weight)

            gen = add_Rt_to_pts(Q, Rt, blended_pose)
            z = gen - y
            new_z = z.reshape(-1, 1)
            new_z = new_z.T @ new_z
            return new_z


        def expr_cost_function(Q, neutral_bar, ids_bar, exprs_bar, id_weight, x, y):
            # x := Q + Rt + expr weight
            r1, r2, r3, tx,ty, tz = x.ravel()[:6]
            
            pose_weight = x[6:, 0]

            Rt = get_Rt(r1,r2,r3, tx,ty,tz)
            blended_pose = get_combine_bar_model(neutral_bar , ids_bar, exprs_bar, id_weight, pose_weight)

            gen = add_Rt_to_pts(Q, Rt, blended_pose)
            z = gen - y
            new_z = z.reshape(-1, 1)
            new_z = new_z.T @ new_z
            return new_z

        def find_camera_matrix(x_3d, x_2d, guessed_projection_Qmat = None):
            # P_{0} @ X - 
            # 12 unknown
            """
                camera matrix P
                [p1 p2  p3  p4 ]   [-p1-]
                [p5 p6  p7  p8 ] = [-p2-]
                [p9 p10 p11 p12]   [-p3-]
                see detail : https://www.cs.cmu.edu/~16385/s17/Slides/11.3_Pose_Estimation.pdf
            """

            def to_homogeneous(x):
                return np.concatenate([x, np.ones((len(x),1))], axis = -1)

            def add_coeff_to_A(mat, x_3d, x_2d, idx):
                r, c = mat.shape
                X = x_3d[idx, :]
                pts_2d = x_2d[idx, :]
                x, y = pts_2d.ravel()
                mat[idx*2, :4] = X
                mat[idx*2+1, 4:8] = X
                mat[idx*2, -4: ] = -x * X
                mat[idx*2+1, -4: ] = -y * X
            x_3d = to_homogeneous(x_3d)
            # setup matrix A
            mat = np.zeros((2*len(x_3d), 12))
            for idx in range(len(x_3d)):
                add_coeff_to_A(mat, x_3d, x_2d, idx)

            u, s, vT = np.linalg.svd(mat)
            sol = vT[-1, :]
            
            
            # cam mat
            mat_P = sol.reshape(3, 4)
            
            xxx = x_3d
            xx = mat_P @ xxx.T
            xx2 = xx/xx[-1,  :]
            x2 = x_2d


            if guessed_projection_Qmat is not None :
                # if we know Q
                Q = guessed_projection_Qmat
                Qinv = np.linalg.inv(Q)
                Rt = Qinv@mat_P
            else:
                # if we don't knonw
                u,s, vT = np.linalg.svd(mat_P)
                        
                # c = vT[-1, :] 
                c = vT[-1, :] 
                c[:] /= c[-1]
                c = c[:-1] #remove hormogeneous to 3d pts

                M = mat_P[:3,:3]
                K, R = scipy.linalg.rq(M)
                Q = K

                #solve flip problem
                if Q[0,0] < 0:
                    Q[0,0] *= -1
                    R[0, :] *= -1
                if Q[1,1] < 0:
                    Q[1,1] *= -1
                    R[1, :] *= -1
                if Q[-1,-1] < 0:
                    Q[:,-1]*=-1
                    R[-1, :] *= -1
                
                # [R, -Rc]
                # Rt = np.concatenate([R, -R@c.reshape(-1,1)[:-1, :]], axis = -1)
                Rt = np.concatenate([R, -R@c.reshape(-1,1)], axis = -1)
            print("gen P : \n", Q@Rt, "\norig : \n",mat_P)
            return Q, Rt


        def coordinate_descent(cost_function, init_x, y, iter_nums, eps = 10e-7):
            if len(init_x.shape) == 1 : 
                init_x = init_x.reshape(-1, 1)

            def cost_f(x):
                return cost_function(x, y)
            
            def cost_grad_wrapper(ind):
                def wrapper(x):
                    copied_x = np.copy(x)
                    copied_x[ind, 0] -= eps
                    f_val = cost_f(copied_x)
                    copied_x[ind, 0] += 2*eps
                    f_h_val = cost_f(copied_x)
                    gradient = (f_h_val - f_val)/(2*eps)
                    gradient_array = np.zeros_like(x)
                    gradient_array[ind, 0 ] = gradient

                    return gradient_array.T         
                def full_grad(x):
                    grad_array = np.zeros_like(x)
                    for i in range(len(x)):
                        copied_x = np.copy(x)
                        copied_x -= eps
                        f_val = cost_f(copied_x)
                        copied_x[i, 0] += eps
                        f_h_val = cost_f(copied_x)
                        gradient = (f_h_val - f_val)/eps*2
                        grad_array[i, 0 ] = gradient
                    return grad_array.T         
                return wrapper, full_grad
            
            x = np.copy(init_x)
            for iter_i in range(iter_nums):
                for i in range(len(x)):
                    f_val = cost_f(x)
                    # x[i, 0] += eps
                    sel_idx_grad_func, full_gradient_func = cost_grad_wrapper(i)
                    coord_grad = sel_idx_grad_func(x).T
                    gradient_direction = full_gradient_func(x).T
 
                    re = opt.line_search(cost_f, sel_idx_grad_func, x, -coord_grad)
                    alpha = re[0]
                    # for safety. when we put too small, and opposite gradient direction into line_search, function will return None,
                    # this if prevent too small gradient.
                    
                    if alpha is None : 
                        alpha = 0
                    
                    x -= coord_grad*alpha

                    x[6:,0] = np.clip(x[6:, 0], 0.0, 1.0)
                    if i in [0,1,2]:
                        x[i] %= np.pi*2
                print("iter : ", iter_i, "i-th of w : ", i,"cost : ", f_val, "\nx", x.ravel(), "alpha : ", alpha, "")
                return x

        iter_num = 10
        # expr_weights = [np.zeros((expr_num, 1)) for _ in range(len(self.img_and_info.keys()))]
        expr_weights = [np.zeros((expr_num, 1)) for _ in range(len(self.img_list))]
        id_weight = np.zeros((id_num, 1))
        time_t = True
        Q_list = [[] for _ in range(len(self.img_list))]
        Rt_list = [[] for _ in range(len(self.img_list))]
        lmk_2d_list = [ info['lmk_data'] for info in self.img_list ]
        
        for i in tqdm.tqdm(range(iter_num)):
            
            for key_id, item in tqdm.tqdm(enumerate(self.img_and_info.values())):
                sel_imgs = [info['img_data'] for info in item ]
                lmk_2ds = np.array([info['lmk_data'] for info in item ])
                index_list =  [ info['index'] for info in item ]
                sel_lmk_idx_list = [ lmk_idx_list[info['index']] for info in item ]

                # nuetral, ids, exprs
                sel_faces = [get_bars(neutral, ids, expr, idx_list) for idx_list in sel_lmk_idx_list]
                
                
                raw_Q_Rt_list = [find_camera_matrix( get_combine_bar_model(face[0], face[1], face[2], id_weight, expr_weights[index]) ,lmk_2d, None) for index, face, lmk_2d in zip(index_list, sel_faces, lmk_2ds) ]

                expr_Q_list = [ Q for Q, _ in raw_Q_Rt_list]
                expr_Rt_list = [ Rt for _, Rt in raw_Q_Rt_list ] 
                
                def exp_cost_builder(Q, neutral, ids, exps):
                    def wrapper(x,y):
                        w = x[:6, 0]
                        weight_norm = w.T@w
                        id_weight = x[6:6+len(ids), :]
                        expr_weight = x[6+len(ids):, :]
                        return default_cost_function(Q, Rt, neutral, ids, exps, id_weight, expr_weight, y ) + weight_norm + (np.sum(w) - 1)**2

                        return expr_cost_function(Q, neutral, ids, exps, id_weight, x,y) + weight_norm + (np.sum(w) - 1)**2
                    return wrapper

                for image_i,( Q, (neutral_, ids_, exprs_), Rt) in enumerate(zip(expr_Q_list, sel_faces, expr_Rt_list)) : 
                    init_weight = np.ones((len(exprs_) + len(ids_)+ 6,1), dtype=np.float64)*0.5
                    r_x, r_y, r_z = decompose_Rt(Rt)
                    tx, ty, tz = Rt[:, -1]
                    init_weight[:6, :] = np.array([r_x, r_y, r_z, tx, ty, tz]).reshape(-1,1)
                    # opt_result = coordinate_descent(exp_cost_builder(Q, neutral_, ids_, exprs_), init_weight, lmk_2ds, 10)

                    # cam_weight = opt_result[:6, 0]
                    # exp_weight = opt_result[len(ids_)+6:, :]
                    # id_weight = opt_result[6:len(ids_)+6, :]
                    # expr_weights[index_list[image_i]] = exp_weight
                    
                    # Rt_list[ index_list[image_i] ] = get_Rt(*cam_weight.ravel())
                    Rt_list[ index_list[image_i] ] =Rt
                    Q_list[ index_list[image_i] ] = expr_Q_list[image_i]

                    path_name = osp.join("cd_test", str(key_id))
                    if not os.path.exists(path_name):
                        os.makedirs(path_name)
                    # vv = get_combine_model(id_weight, exp_weight)
                    vv = get_combine_model(id_weight, np.zeros_like(expr_weights[0]))
                    igl.write_triangle_mesh(os.path.join(path_name, "exp_Rt_opt_" +str(image_i)+ ".obj"), vv, self.neutral_mesh_f)
                    cv2.imwrite(osp.join(path_name, "exp_Rt_opt_" +str(image_i)+ ".png"), sel_imgs[image_i])
                    cv2.waitKey(0)

                # expr_weights[key_id] = exp_weight
            # #===========================================================================================
            #     # expr_weights.append(exp_weight)
                for index in index_list:
                    # time_t = draw_cv(index, time_t, id_weight, expr_weights, cam_scales, cam_rots, cam_tvecs)
                    time_t = draw_cv(index,key_id, time_t, id_weight, expr_weights, Q_list, Rt_list)
                    # path_name = osp.join("testdir", str(index))
                    # if not os.path.exists(path_name):
                        # os.makedirs(path_name)
                    # vv = get_combine_model(id_weight, exp_weight)
                    # igl.write_triangle_mesh(os.path.join(path_name, "opt1_iter_num_" +str(i)+ ".obj"), vv, self.neutral_mesh_f)
            

            # phase 2
            def id_cost_funciton_builder(Q_list, Rt_list, v_idx_sel_index_list, expr_weight_list, lmk_2d_list):
                
                assert(Q_list != Rt_list, "Q_list, Rt_list is not same")
                prp_vert = [] 
                for v_idx_sel_index in v_idx_sel_index_list:
                    new_v = get_bars(neutral, ids, expr, v_idx_sel_index)
                    prp_vert.append(new_v)
                def wrapper(x,y):# we do not use y in here.
                    cost_z = 0
                    for Q, Rt, (neutral_b, ids_b, expr_b), expr_w, lmk_2d in zip(Q_list, Rt_list, prp_vert, expr_weight_list, lmk_2d_list):
                        cost_z += default_cost_function(Q, Rt, neutral_b, ids_b, expr_b, x, expr_w, lmk_2d) 
                    return cost_z
                return wrapper

            # res_id_weight = coordinate_descent(id_cost_funciton_builder(Q_list, Rt_list, lmk_idx_list, expr_weights, lmk_2d_list), id_weight, None, 10)
            # id_weight  = res_id_weight

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
