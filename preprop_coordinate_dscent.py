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

        def coordinate_descent(cost_function, Q, init_x, y, iter_num, eps = 10e-7):
            if len(init_x.shape) == 1 : 
                init_x = init_x.reshape(-1, 1)
            def cost_wrapper(x):
                    cons = x[6:, :] # regularization term
                    return cost_function(Q, neutral, pose, x, y) + cons.T@cons 
            
            def cost_grad_wrapper(ind):
                def wrapper(x):
                    copied_x = np.copy(x)
                    copied_x[ind, 0] -= eps
                    f_val = cost_wrapper(copied_x)
                    copied_x[ind, 0] += 2*eps
                    f_h_val = cost_wrapper(copied_x)
                    gradient = (f_h_val - f_val)/(2*eps)
                    gradient_array = np.zeros_like(x)
                    gradient_array[ind, 0 ] = gradient

                    return gradient_array.T         
                def full_grad(x):
                    grad_array = np.zeros_like(x)
                    for i in range(len(x)):
                        copied_x = np.copy(x)
                        copied_x -= eps
                        f_val = cost_wrapper(copied_x)
                        copied_x[i, 0] += eps
                        f_h_val = cost_wrapper(copied_x)
                        gradient = (f_h_val - f_val)/eps*2
                        grad_array[i, 0 ] = gradient
                    return grad_array.T         
                return wrapper, full_grad
            
            x = np.copy(init_x)
            for iter_i in range(iter_nums):
                for i in range(len(x)):
                    f_val = cost_wrapper(x)
                    # x[i, 0] += eps
                    sel_idx_grad_func, full_gradient_func = cost_grad_wrapper(i)
                    coord_grad = sel_idx_grad_func(x).T
                    gradient_direction = full_gradient_func(x).T
                    # if np.abs(coord_grad[i]) < 1.88e-6: # if too small gradient value, line search can't find appropriate alpha.(they return None...)
                        # continue
                    # f_val_h = cost_function(neutral, x, y)
                    # f_grad = (f_val_h-f_val)/eps
                    # x[i, 0] -= eps
                    re = opt.line_search(cost_wrapper, sel_idx_grad_func, x, -coord_grad)
                    # re = opt.line_search(cost_wrapper, sel_idx_grad_func, x, -gradient_direction)
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

