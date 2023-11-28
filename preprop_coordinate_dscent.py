#
import cv2, glob, os 
import os.path as osp
import numpy as np 
import geo_func as geo 
import igl 
import scipy.optimize as opt
import re
import argparse
import tqdm
import scipy.spatial as sp
import camera_clib as clib
import data_loader as dl
import open3d as o3d
import open3d.visualization.rendering as rendering
import visualizer as vis 
from collections import Counter
import scipy 
import yaml
import scipy.optimize as opt
import copy 


import argparse

np.set_printoptions(precision=3, suppress=True)

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
        self.f = self.neutral_mesh_f
        expr_paths = glob.glob(osp.join(root_dir, "shapes", "generic_neutral_mesh", "**.obj"))
        expr_paths.sort(key= natural_keys)
        print(expr_paths)
        mouth_name_list = [t for t in expr_paths if osp.basename(t).startswith("mouth")]
        self.mouth_coeff_index = [expr_paths.index(mouse_name) for mouse_name in mouth_name_list ]
        self.mouth_names = [osp.basename(m) for m in mouth_name_list]
        mesh_collect = []
        for expr_path in tqdm.tqdm(expr_paths, leave=False):
            v, f = igl.read_triangle_mesh(expr_path)
            mesh_collect.append(v)
        self.meshes.append(mesh_collect)
        self.expr_meshes = mesh_collect



        # meta info extended mesh 
        # expr_index_list = [(np.linalg.norm((self.neutral_mesh_v - m).reshape(-1,1)), i) for i,m in enumerate(self.expr_meshes)]
        # sorted_expr_index_list =sorted(expr_index_list, key=lambda x:x[0], reverse=True)
        # self.norm_base_expr_mesh_index1 = [x[-1] for x in sorted_expr_index_list]
        expr_index_list = [(np.linalg.norm((self.neutral_mesh_v - m)[lmk_idx].reshape(-1,1)), i) for i,m in enumerate(self.expr_meshes)]
        sorted_expr_index_list =sorted(expr_index_list, key=lambda x:x[0], reverse=True)
        self.norm_base_expr_mesh_index = [x[-1] for x in sorted_expr_index_list]
        reverse = [ (i, x[-1]) for i, x in enumerate(sorted_expr_index_list)]
        reverse = sorted(reverse, key=lambda x:x[-1])
        self.norm_base_expr_mesh_index_revert = [x[0] for x in reverse]


            



        print("neutral boundary calc start.")
        halfedge = vis.EdgeFace(self.neutral_mesh_f)
        halfedge.build()
        boundary_list = halfedge.get_boundary_v_list()
        boundary_list_size = [len(bb) for bb in boundary_list]
        max_idx = 0
        val = 0
        for i, bb in enumerate(boundary_list_size):
            if val < bb:
                val = bb
                max_idx = i
        self.mesh_boundary_index = boundary_list[max_idx]
        print("neutral boundary calc was done.")

            



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
            category_predef_weight = np.load(osp.join("./predefined_face_weight", key + ".npy"))

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
                img_data = {'category_index' : category_idx, 'index' : len(self.img_list), "name" : name, "lmk_data" : lmk_data, "img_data": img_data, "predef_weight" : category_predef_weight}
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
    


        
    def convert_focal_length(w, h, mm_focal_length=36):
        """
        mm to pixel
        """
        m = max(h, w)
        standard_sensor_size = 36
        m *mm_focal_length / standard_sensor_size
    
    def get_combine_bar_model(self, neutral_bar, ids_bar=None, expr_bar=None, w_i=None, w_e=None ):
        v_size, dim = neutral_bar.shape
        res = np.zeros((v_size*dim, 1), dtype=np.float32)
        if ids_bar is not None and w_i is not None:
            id_num, id_v_size, id_dim = ids_bar.shape 
            reshaped_id = ids_bar.reshape(id_num, id_v_size*id_dim).T
            res += reshaped_id@w_i.reshape(-1,1) 

        if expr_bar is not None or w_e is not None:
            expr_num, expr_v_size, expr_dim = expr_bar.shape
            reshaped_expr = expr_bar.reshape(expr_num, expr_v_size*expr_dim).T
            res += reshaped_expr@w_e.reshape(-1,1)

        res = res.reshape(v_size, dim)
        return neutral_bar + res
    
    def get_combine_model(self, neutral, ids=None, expr=None, w_i=None, w_e=None):
        
        v_size, dim = neutral.shape
        res = np.zeros((v_size*dim, 1), dtype=np.float32)
        if ids is not None or w_i is not None :
            id_num, id_v_size, id_dim = ids.shape
            new_ids = ids.reshape(id_num, id_v_size*id_dim).T
            res +=  new_ids@w_i

        if expr is not None or w_e is not  None :
            expr_num, expr_v_size, expr_dim = expr.shape
            new_exps = expr.reshape(expr_num, expr_v_size*expr_dim ).T
            res += new_exps@w_e
        res = res.reshape(v_size, dim)
        return neutral + res
    
    def get_bars(self, neutral, ids, exprs, sel_lmk_idx):
        neutral_bar = neutral[sel_lmk_idx, :]
        ids_bar = ids[:, sel_lmk_idx, :]
        expr_bar = exprs[:, sel_lmk_idx, :]
        return neutral_bar, ids_bar, expr_bar

    def default_cost_function(self, Q, Rt, neutral_bar, ids_bar, exprs_bar, id_weight, expr_weight, y):
        # x := Q + Rt + expr weight

        blended_pose = self.get_combine_bar_model(neutral_bar , ids_bar, exprs_bar, id_weight, expr_weight)
        
        gen = self.add_Rt_to_pts(Q, Rt, blended_pose)
        z = gen - y
        new_z = z.reshape(-1, 1)
        new_z = new_z.T @ new_z
        return new_z
    
    def get_Rt(self, theta_x, theta_y, theta_z, tx, ty, tz):
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
    
    def add_Rt_to_pts(self, Q, Rt, x):
            R = Rt[:3,:3]
            t = Rt[:, -1, None]
            xt = x.T
            Rx = R @ xt 
            Rxt = Rx+t
            pj_Rxt = Q @ Rxt
            res = pj_Rxt/pj_Rxt[-1, :]
            return res[:2, :].T
            
    def decompose_Rt(self, Rt):
        """
        input : 
            Z,y,x sequentially
        return 
            x, y, z angle
        see also https://en.wikipedia.org/wiki/Rotation_matrix
        """
        y_angle = np.arctan2(-1*Rt[2,0],np.sqrt(Rt[0,0]**2+Rt[1,0]**2))
        x_angle = np.arctan2(Rt[2,1]/np.cos(y_angle),Rt[2,2]/np.cos(y_angle))
        z_angle = np.arctan2(Rt[1,0]/np.cos(y_angle),Rt[0,0]/np.cos(y_angle))


        return x_angle, y_angle, z_angle
    
    def find_camera_matrix(self, x_3d, x_2d, guessed_projection_Qmat = None):
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
        mat = mat.T @mat
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
                    
            c = vT[-1, :] 
            # c[:] /= c[-1]
            # c = c[:-1] #remove hormogeneous to 3d pts

            M = mat_P[:3,:3]
            K, R = scipy.linalg.rq(M)
            testK1 = K
            testR1 = R
            Q = K
            Q/=K[-1,-1]
            
            # U, S, Vt = np.linalg.svd(M)
            # R = np.dot(Vt.T,U.T)
            
            # # special reflection case
            # eye = np.eye(3, dtype=np.float32)
            # d = np.linalg.det(R)
            # if d < 0:
            #     print("Reflection detected")
            # eye[2,2]= d
            # R = np.dot(np.dot(Vt.T, eye),U.T)
            # K = M@R.T

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
            
        #     # [R, -Rc]
        #     # Rt = np.concatenate([R, -R@c.reshape(-1,1)], axis = -1)
        #     RR = np.identity(4)
        #     RR[:3,:3] = R
        #     Mc = RR@c.reshape(-1,1)
        #     Mc[:,:] /= Mc[-1, :]
        #     Mc = Mc[:-1, :]
        #     Rt = np.concatenate([R, -Mc], axis = -1)
        # print("gen P : \n", Q@Rt, "\norig : \n",mat_P)
        # test =Q@Rt


        c = vT[-1, :] 
        c[:] /= c[-1]
        c = c[:-1] #remove hormogeneous to 3d pts
        Q = K
        Rt= np.concatenate([R, -R@c.reshape(-1,1)], axis=-1)
        # print("====")
        # print(mat_P)
        # print(K@Rt)

        return Q, Rt
    def save_png(self, root_path, file_name, neutral, id, expr, id_weight, expr_weight, Q, Rt, img, lmk2d, iteration, **kwargs):
        if not osp.exists(root_path):
            os.makedirs(root_path)

        vv = self.get_combine_model(neutral, id, expr, id_weight, expr_weight)
        pts2d = self.add_Rt_to_pts(Q, Rt, vv)
        # time_t = draw_cv(index, time_t, id_weight, expr_weights, cam_scales, cam_rots, cam_tvecs)
        contour = self.contour['full_index']

        gt_lmk_img = vis.draw_pts(img, np.array(lmk2d), color=(0,0,255), width=1000, caption = "Ground Truth Landmark")

        pred_lmk_img = vis.draw_pts(img, pts2d[lmk_idx], color=(0,0,255), width=1000, caption = "Fitting Landmark")
        
        pred_pts_img = vis.draw_pts(img, pts2d, color=(0,0,255), width = 1000, radius = 1, caption = "Fitting Landmark : iteration : "+str(iteration))
        # new_cont_img = vis.draw_pts(img, new_contour, color=(0,0,255))
        new_cont_img = vis.draw_contour(img, pts2d, np.array(lmk_idx)[contour], color=(0,0,255), line_color=(0,255,0), caption=" ")
        gt_lmk_img = vis.draw_pts(new_cont_img, np.array(lmk2d)[self.contour['full_index']], color=(0,255,255))
        
        mesh_contour_img = vis.draw_contour(gt_lmk_img, pts2d, self.mesh_boundary_index, color=(255,0,0), width =1000, caption = "Contour Landmark Selection based on Covexhull")
        
        vis.set_delay(1)


        mesh_overlay_image = vis.draw_mesh_to_img(img, Q, Rt, vv, self.f, (1.0, 0, 0), width=1000, caption = "Result Mesh Overlay")
            
        inner_face_lmk_idx = self.mouse, self.eyebrow, self.nose, self.eye
        inner_face_lmk_idx = self.mouse['full_index'] + self.eyebrow['left_eyebrow']['full_index'] + \
            self.eyebrow['right_eyebrow']['full_index'] + self.nose['vertical'] + self.nose['horizontal']+\
            self.eye['left_eye']['full_index'] + self.eye['right_eye']['full_index']
        #########################################################
        #draw inner shapes
        pred_lmk_img = vis.draw_pts_mapping(img, pts2d[np.array(lmk_idx)[inner_face_lmk_idx]],np.array(lmk2d)[inner_face_lmk_idx], color=(255,0,0))
        gt_lmk_img = vis.draw_pts(pred_lmk_img, np.array(lmk2d)[inner_face_lmk_idx], color=(0,255,255))
        pred_lmk_img = vis.draw_pts(gt_lmk_img, pts2d[np.array(lmk_idx)[inner_face_lmk_idx]],color=(0,0,255),width=1000, caption = "Mapping Landmark")
        # checking cors mapping 
        alpha = kwargs.get("alpha", None)

        ip = [pred_pts_img, mesh_contour_img, pred_lmk_img, mesh_overlay_image]
        if alpha is not None:
            vv = self.get_combine_model(neutral, id, expr, id_weight, alpha)
            rest = vis.draw_mesh_to_img(img, Q, Rt, vv, self.f, (1.0, 0, 0), width=1000, caption = "Const & init val alpha* Mesh Overlay")
            ip.append(rest)



        import math 
        
        concat_img = vis.concatenate_img(2, math.ceil(len(ip)/2), *ip )
        vis.save(osp.join(root_path, file_name + ".png"), concat_img)
        concat_img = vis.resize_img( concat_img, 1000)
        vis.show("test", concat_img )

    def find_contour(self, lmk2d, proj_3d_v):
        
        mask_indices = [ii for ii in  range(len(proj_3d_v)) if ii not in  self.unconcern_mesh_idx]

        full_version = proj_3d_v
        proj_3d_v = np.zeros((len(full_version) - len(self.unconcern_mesh_idx), 2))
        mapper = np.zeros((len(proj_3d_v)), dtype = np.uint32)
        # proj_3d_v = proj_3d_v[mask_indices, :]
        idx = 0
        self.contour_pts_idx
        for ii in range(len(full_version)):
            if ii not in self.unconcern_mesh_idx:
                proj_3d_v[idx, :] = full_version[ii, :]
                mapper[idx] = ii
                idx += 1
            

        hull = sp.ConvexHull(proj_3d_v, qhull_options="QJ")
        convex_index = hull.vertices
        # import matplotlib.pyplot as plt 
        # plt.plot(proj_3d_v[:, 0], proj_3d_v[:, 1 ], 'gx')
        # plt.plot(proj_3d_v[convex_index, 0], proj_3d_v[convex_index, 1 ], 'bo')
        # plt.xlim(0, 5000)
        # plt.ylim(0, 3500)
        # plt.show()
        kd = sp.cKDTree(proj_3d_v[convex_index, :])
        d, idx = kd.query(lmk2d)
        return mapper[convex_index[idx]]
        # return convex_index[idx]
    
    
    

    def coordinate_descent_LBFGS(self, cost_function, init_x, y, iter_nums, eps = 10e-7, clip_func = None, camera_refinement_func = None ,  contour_mapper_func = None, **kwargs):
        """
            coordinate descent by LBFGS(super duper)
            si sisisisisisisisi

            kwrgs:
                verbose(bool) : render img
                img : image
                lmk_idx : landmark index of mesh
                contour_index : contour index part of lmk_idx
                mesh : {color, Q, neutral, exprs, f}
                y_info : {pts : color , line : color }
                x_info : {pts : color line : color }
                blend_function(x) 
                width(optional) : default is 1000
                


        """

        def verbose_function( w, **options):
            """
            "prev_neutral" 
            "prev_exprs"
            "prev_lmk_idx"
            """
            Rt= self.get_Rt(*w[-6:, :].ravel())
            w = w[:-6, :]

            img = kwargs.get("img", None)
            mesh = kwargs.get("mesh", None)
            if img is not None :
                Q = mesh.get("Q", None )
                neutral = mesh.get("neutral", None)
                exprs = mesh.get("exprs", None)
                f = mesh.get("f", None)
                m_color = mesh.get("color", (0.5, 0, 0))
                pts_3d = self.get_combine_model(neutral, None, exprs,None, w )
                pts_2d = self.add_Rt_to_pts(Q, Rt, pts_3d)

                if not (Q is None or  neutral is None or exprs is None or f is None ): 
                    res = vis.draw_mesh_to_img(img, Q, Rt, pts_3d, f, color = m_color)


                # 
                lmk_idx = kwargs.get("lmk_idx", None )
                contour_index = kwargs.get("contour_index", None )
                width = kwargs.get("width", 1000)
                
                x_info = kwargs.get("x_info", None)
                y_info = kwargs.get("y_info", None)

                prev_lmk_idx = options.get("prev_lmk_idx", None )
                if not (lmk_idx is None or  contour_index is None) :
                    print(prev_lmk_idx == lmk_idx)
                    if  prev_lmk_idx is not None:
                        pts_color = x_info.get("pts", (255,0,0))
                        prev_pts3d = self.get_combine_model(neutral, None, exprs,None, w )
                        prev_pts_2d = self.add_Rt_to_pts(Q, Rt, prev_pts3d)
                        res = vis.draw_contour(res, prev_pts_2d[prev_lmk_idx, :], contour_index, (255,0,255), (255,0,255))
                        if x_info is not None :
                            res = vis.draw_pts_mapping(res, pts_2d[lmk_idx, :][contour_index],  prev_pts_2d[prev_lmk_idx, :][contour_index], color=(255,255,0))
                    if x_info is not None:
                        pts_color = x_info.get("pts", (255,0,0))
                        line_color = x_info.get("line", (255,255,0))
                        res = vis.draw_contour(res, pts_2d[lmk_idx, :], contour_index, pts_color, line_color)
                    if y_info is not None:
                        pts_color = y_info.get("pts", (0,0,255))
                        line_color = y_info.get("line", (0,255,255))
                        res = vis.draw_contour(res, y, contour_index, pts_color, line_color)
                    if x_info is not None and y_info is not None: 
                        res = vis.draw_pts_mapping(res, pts_2d[lmk_idx, :][contour_index], y[contour_index])

                        
                    vis.set_delay(1)

                    res = vis.resize_img(res, width)
                    vis.show("shu shu, ", res)

        if len(init_x.shape) == 1 : 
            init_x = init_x.reshape(-1, 1)
        
        def cost_f(x):
            return cost_function(x, y)
        
        def cost_function_builder(ind, orig_x):
            def wrapper(x):
                copied_x = np.copy(orig_x)
                copied_x[ind, 0] = x
                return cost_f(copied_x)
            return wrapper
        
        def cost_grad_wrapper(ind, orig_x):
            def wrapper(x):
                copied_x = np.copy(orig_x)
                copied_x[ind, 0] = x - eps
                f_val = cost_f(copied_x)
                copied_x = np.copy(orig_x)
                copied_x[ind, 0] = x + eps
                f_h_val = cost_f(copied_x)
                gradient = (f_h_val - f_val)/(2*eps)
                return  gradient
            def full_grad(x):
                grad_array = np.zeros_like(x)
                for i in range(len(x)):
                    copied_x = np.copy(x)
                    copied_x -= eps
                    f_val = cost_f(copied_x)
                    copied_x += eps
                    f_h_val = cost_f(copied_x)
                    gradient = (f_h_val - f_val)/eps*2
                    grad_array[i, 0 ] = gradient
                return grad_array.T         
            return wrapper, full_grad
        
        x = np.copy(init_x)
        grad_history = [np.zeros_like(x) for i in range(iter_nums)]
        alpah_history = [np.zeros_like(x) for i in range(iter_nums)]
        prev_f_val = 9999999999999

        options = {'disp':False}

        w_bounds = [(0.0, 1.0) ]
        rot_bounds = [(0.0, np.pi*2)]


        new_lmk_idx_history = [ ]
        print(kwargs.keys())
        start_lmk_idx = kwargs.get("lmk_idx", None)
        if  start_lmk_idx is not None:
            start_lmk_idx = copy.deepcopy( start_lmk_idx)
            new_lmk_idx_history.append(start_lmk_idx)


        for iter_i in range(iter_nums):
            
            slide = None
            if camera_refinement_func is not None :
                slide = -6
            
                if iter_i != 0 :
            
                    rx, ry, rz, tx, ty, tz = camera_refinement_func(x[:slide], y)
                    # print(x[slide:, :].reshape(-1))
                    # print(np.array([rx, ry, rz, tx, ty, tz]).reshape(-1))
                    x[slide:, :] = np.array([rx, ry, rz, tx, ty, tz]).reshape(-1,1)

                    if contour_mapper_func is not None :
                        new_lmk_indx = contour_mapper_func(x) # do not modify its values.
                        new_lmk_idx_history.append(new_lmk_indx)
                else:
                    rx, ry, rz, tx, ty, tz = camera_refinement_func(x[:slide], y, True)
                    # print(x[slide:, :].reshape(-1))
                    # print(np.array([rx, ry, rz, tx, ty, tz]).reshape(-1))
                    x[slide:, :] = np.array([rx, ry, rz, tx, ty, tz]).reshape(-1,1)

                    if contour_mapper_func is not None :
                        new_lmk_indx = contour_mapper_func(x) # do not modify its values.
                        new_lmk_idx_history.append(new_lmk_indx)
            
            for i in range(len(x[:slide])):
                xi = x[i, 0]
                
                sel_ind_cost_f = cost_function_builder(i, x)
                
                f_val = sel_ind_cost_f(xi)
                sel_idx_grad_func, _ = cost_grad_wrapper(i, x)
                
                coord_grad = sel_idx_grad_func(xi).T

                # opt.minimize(fun=cost_f, x0 = init_x, jac= ,method="BFGS", options=options)
                res = opt.minimize(fun=sel_ind_cost_f, x0 = np.array([[xi]]), jac = sel_idx_grad_func, method="L-BFGS-B", bounds = w_bounds, options=options)
                x[i, :] = res.x

                # res = opt.minimize(fun=sel_ind_cost_f, x0 = np.array([[xi]]), jac = sel_idx_grad_func, method="L-BFGS-B", bounds = bounds,options=options)
                # x = clip_func(x)




            if kwargs.get("verbose", False ) :
                if len(new_lmk_idx_history) == 1:
                    verbose_function(x, **{"prev_lmk_idx" :new_lmk_idx_history[-1]})
                else:
                    verbose_function(x, **{"prev_lmk_idx" :new_lmk_idx_history[-2]})
                    



            if abs(f_val - prev_f_val) < 10 :
            # if np.all( np.abs(grad_history[iter_i]) < 10e-7, axis=0):
                print("stopped at iteration : ", iter_i, ". all gradient is closed to zero, stop optimizing.")
                break
            
            if not (f_val < prev_f_val): 
                print("fval is greater than prev_f_val")
                break


            prev_f_val = f_val
            # print("iter : ", iter_i, "cost : ", f_val, "\nx", x.ravel())
            print("iter : ", iter_i, "cost : ", f_val, "grad mean : ", np.mean(grad_history[iter_i]))
        return x, grad_history, alpah_history

    def find_camera_matrix(self, x_3d, x_2d, guessed_projection_Qmat = None):
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
        mat = mat.T @mat
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
                    
            c = vT[-1, :] 
            # c[:] /= c[-1]
            # c = c[:-1] #remove hormogeneous to 3d pts

            M = mat_P[:3,:3]
            K, R = scipy.linalg.rq(M)
            testK1 = K
            testR1 = R
            Q = K
            Q/=K[-1,-1]
            
            # U, S, Vt = np.linalg.svd(M)
            # R = np.dot(Vt.T,U.T)
            
            # # special reflection case
            # eye = np.eye(3, dtype=np.float32)
            # d = np.linalg.det(R)
            # if d < 0:
            #     print("Reflection detected")
            # eye[2,2]= d
            # R = np.dot(np.dot(Vt.T, eye),U.T)
            # K = M@R.T

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
            
        #     # [R, -Rc]
        #     # Rt = np.concatenate([R, -R@c.reshape(-1,1)], axis = -1)
        #     RR = np.identity(4)
        #     RR[:3,:3] = R
        #     Mc = RR@c.reshape(-1,1)
        #     Mc[:,:] /= Mc[-1, :]
        #     Mc = Mc[:-1, :]
        #     Rt = np.concatenate([R, -Mc], axis = -1)
        # print("gen P : \n", Q@Rt, "\norig : \n",mat_P)
        # test =Q@Rt


            c = vT[-1, :] 
            c[:] /= c[-1]
            c = c[:-1] #remove hormogeneous to 3d pts
            Q = K
            Rt= np.concatenate([R, -R@c.reshape(-1,1)], axis=-1)
            # print("====")
            # print(mat_P)
            # print(K@Rt)

        return Q, Rt

    def coordinate_descent(self, cost_function, init_x, y, iter_nums, eps = 10e-7, clip_func = None ):
        if len(init_x.shape) == 1 : 
            init_x = init_x.reshape(-1, 1)

        def cost_f(x):
            return cost_function(x, y)
        
        def cost_grad_wrapper(ind):
            def wrapper(x):
                copied_x = np.copy(x)
                copied_x[ind, 0] -= eps
                f_val = cost_f(copied_x)
                copied_x = np.copy(x)
                copied_x[ind, 0] += eps
                # copied_x[ind, 0] += 2*eps
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
        grad_history = [np.zeros_like(x) for i in range(iter_nums)]
        alpah_history = [np.zeros_like(x) for i in range(iter_nums)]
        prev_f_val = 9999999999999
        for iter_i in range(iter_nums):
            
            

            for i in range(len(x)):
                f_val = cost_f(x)
                sel_idx_grad_func, _ = cost_grad_wrapper(i)
                coord_grad = sel_idx_grad_func(x).T

                grad_history[iter_i][i, :] = coord_grad[i, :]
                re = opt.line_search(cost_f, sel_idx_grad_func, x, -coord_grad)
                alpha = re[0]
                # for safety. when we put too small, and opposite gradient direction into line_search, function will return None,
                # this if prevent too small gradient.
                


                if alpha is None : 
                    alpha = 0
                alpah_history[iter_i][i, :] = alpha
                x -= coord_grad*alpha
                x = clip_func(x)
            
            if abs(f_val - prev_f_val) < 10:
            # if np.all( np.abs(grad_history[iter_i]) < 10e-7, axis=0):
                print("stopped at iteration : ", iter_i, ". all gradient is closed to zero, stop optimizing.")
                break
            prev_f_val = f_val
            # print("iter : ", iter_i, "cost : ", f_val, "\nx", x.ravel())
            print("iter : ", iter_i, "cost : ", f_val, "grad mean : ", np.mean(grad_history[iter_i]))
        return x, grad_history, alpah_history
    
    # def shape_fit(self, lmks_2d, images, id_meshes, expr_meshes, lmk_idx):
    def shape_fit(self, id_meshes, expr_meshes, lmk_idx, force_recalcutaion = False):
        # Q = clib.calibrate('./images/checker4/*.jpg')
        # extract actor-specific blendshapes
        # iteratively fit shapes
        # it consists with 2 phases.
        # we find id weight, expr weight, proj matrix per each iteration, 
        # but we don't save and reuse each weights,
        # then why we should iterate optimization, 
        # The main reason why we calculate weights is find contour points and fit them to real face.
        # 
        
        if not force_recalcutaion:
            if osp.exists("./cd_test/identity_weight.txt.npy"):
                return 


        self.unconcern_mesh_idx = np.load("./unconcerned_pts.npy")
        self.contour_pts_idx = np.load("./contour_pts.npy")
        lmk_idx = np.array(lmk_idx)
        lmk_idx_list = np.stack([lmk_idx for _ in range(len(self.img_list))],axis=0)

        neutral = self.neutral_mesh_v
        # neutral_bar = neutral[lmk_idx, :]

        ids_meshes = np.array(id_meshes)
        ids = ids_meshes - np.expand_dims(neutral, axis=0)
        # ids_bar = ids[..., lmk_idx, :]
        id_num,_,_ = ids.shape


        expr_meshes = np.array(expr_meshes)
        expr = expr_meshes - np.expand_dims(neutral, axis=0)
        # expr_bar = expr[..., lmk_idx, :]
        expr_num, _,_ = expr.shape


        
        inner_face_lmk_idx = self.mouse, self.eyebrow, self.nose, self.eye
        inner_face_lmk_idx = self.mouse['full_index'] + self.eyebrow['left_eyebrow']['full_index'] + \
            self.eyebrow['right_eyebrow']['full_index'] + self.nose['vertical'] + self.nose['horizontal']+\
            self.eye['left_eye']['full_index'] + self.eye['right_eye']['full_index']
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
                Z,y,x sequentially
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

        # def default_cost_function(Q, Rt, neutral_bar, ids_bar, exprs_bar, id_weight, expr_weight, y):
            # Q @ neutral_bar
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
            mat = mat.T @mat
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
                        
                c = vT[-1, :] 
                # c[:] /= c[-1]
                # c = c[:-1] #remove hormogeneous to 3d pts

                M = mat_P[:3,:3]
                K, R = scipy.linalg.rq(M)
                testK1 = K
                testR1 = R
                Q = K
                Q/=K[-1,-1]
                
                # U, S, Vt = np.linalg.svd(M)
                # R = np.dot(Vt.T,U.T)
                
                # # special reflection case
                # eye = np.eye(3, dtype=np.float32)
                # d = np.linalg.det(R)
                # if d < 0:
                #     print("Reflection detected")
                # eye[2,2]= d
                # R = np.dot(np.dot(Vt.T, eye),U.T)
                # K = M@R.T

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
                
            #     # [R, -Rc]
            #     # Rt = np.concatenate([R, -R@c.reshape(-1,1)], axis = -1)
            #     RR = np.identity(4)
            #     RR[:3,:3] = R
            #     Mc = RR@c.reshape(-1,1)
            #     Mc[:,:] /= Mc[-1, :]
            #     Mc = Mc[:-1, :]
            #     Rt = np.concatenate([R, -Mc], axis = -1)
            # print("gen P : \n", Q@Rt, "\norig : \n",mat_P)
            # test =Q@Rt


            c = vT[-1, :] 
            c[:] /= c[-1]
            c = c[:-1] #remove hormogeneous to 3d pts
            Q = K
            Rt= np.concatenate([R, -R@c.reshape(-1,1)], axis=-1)
            # print("====")
            # print(mat_P)
            # print(K@Rt)

            return Q, Rt


        def coordinate_descent(cost_function, init_x, y, iter_nums, eps = 10e-7, clip_func = None ):
            if len(init_x.shape) == 1 : 
                init_x = init_x.reshape(-1, 1)

            def cost_f(x):
                return cost_function(x, y)
            
            def cost_grad_wrapper(ind):
                def wrapper(x):
                    copied_x = np.copy(x)
                    copied_x[ind, 0] -= eps
                    f_val = cost_f(copied_x)
                    copied_x = np.copy(x)
                    copied_x[ind, 0] += eps
                    # copied_x[ind, 0] += 2*eps
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
            grad_history = [np.zeros_like(x) for i in range(iter_nums)]
            alpah_history = [np.zeros_like(x) for i in range(iter_nums)]
            prev_f_val = 9999999999999
            for iter_i in range(iter_nums):
                
                

                for i in range(len(x)):
                    f_val = cost_f(x)
                    sel_idx_grad_func, _ = cost_grad_wrapper(i)
                    coord_grad = sel_idx_grad_func(x).T

                    grad_history[iter_i][i, :] = coord_grad[i, :]
                    re = opt.line_search(cost_f, sel_idx_grad_func, x, -coord_grad)
                    alpha = re[0]
                    # for safety. when we put too small, and opposite gradient direction into line_search, function will return None,
                    # this if prevent too small gradient.
                    


                    if alpha is None : 
                        alpha = 0
                    alpah_history[iter_i][i, :] = alpha
                    x -= coord_grad*alpha
                    x = clip_func(x)
                
                if abs(f_val - prev_f_val) < 10:
                # if np.all( np.abs(grad_history[iter_i]) < 10e-7, axis=0):
                    print("stopped at iteration : ", iter_i, ". all gradient is closed to zero, stop optimizing.")
                    break
                prev_f_val = f_val
                # print("iter : ", iter_i, "cost : ", f_val, "\nx", x.ravel())
                print("iter : ", iter_i, "cost : ", f_val, "grad mean : ", np.mean(grad_history[iter_i]))
            return x, grad_history, alpah_history
        
        
        
        def grad_descent(cost_function, init_x, y, iter_nums, eps = 10e-7):
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
            for i in range(30):
                f_val = cost_f(x)
                # x[i, 0] += eps
                sel_idx_grad_func, full_gradient_func = cost_grad_wrapper(i)
                coord_grad = sel_idx_grad_func(x).T
                gradient_direction = full_gradient_func(x).T

                re = opt.line_search(cost_f, full_gradient_func, x, -gradient_direction)
                alpha = re[0]
                # for safety. when we put too small, and opposite gradient direction into line_search, function will return None,
                # this if prevent too small gradient.
                
                if alpha is None : 
                    alpha = 0.000001
                
                x -= gradient_direction*alpha
                x = np.clip(x, 0.0, None)
                # x[6:,0] = np.clip(x[6:, 0], 0.0, 1.0)
                # if i in [0,1,2]:
                #     x[i] %= np.pi*2
            print("cost : ", f_val, "\nx", x.ravel(), "\n alpha : ", alpha, "\ngrad : ", gradient_direction)
            return x

        def bfgs_method(cost_function, init_x, y, iter_nums, eps = 10e-7):
            if len(init_x.shape) == 1 : 
                init_x = init_x.reshape(-1, 1)

            def cost_f(x):
                x = x.reshape(-1,1)
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
                    grad_array = np.zeros_like(x).reshape(-1,1)
                    for i in range(len(x)):
                        copied_x = np.copy(x)
                        copied_x = copied_x.reshape(-1,1)
                        copied_x -= eps
                        f_val = cost_f(copied_x)
                        copied_x[i, 0] += eps
                        f_h_val = cost_f(copied_x)
                        gradient = (f_h_val - f_val)/eps*2
                        grad_array[i, 0 ] = gradient
                    return grad_array.reshape(-1)        
                return wrapper, full_grad
            x = np.copy(init_x)
            import scipy.optimize as opt 

            sel_idx_grad_func, full_gradient_func = cost_grad_wrapper(0)
            res = opt.minimize(cost_f, init_x, method="BFGS", jac=full_gradient_func, bounds=(-1,1), options={'maxiter':100})
            print(res.x)
            print(res.message)
            return res.x.reshape(-1,1)
        


        def find_contour(lmk2d, proj_3d_v):
            return self.find_contour(lmk2d, proj_3d_v)
            hull = sp.ConvexHull(proj_3d_v)
            convex_index = hull.vertices
            kd = sp.cKDTree(proj_3d_v[convex_index, :])
            d, idx = kd.query(lmk2d)
            return convex_index[idx]


        def save_png(root_path, postfix, *index_list, **meta_param):
            if not osp.exists(root_path):
                os.makedirs(root_path)
            iter_str = meta_param.get("iteration","")

            for index in index_list:
                img = self.img_list[index]['img_data']
                name = self.img_list[index]['name']
                vv = get_combine_model(id_weight, expr_weights[index])
                pts2d = add_Rt_to_pts(Q_list[index], Rt_list[index], vv)
                # time_t = draw_cv(index, time_t, id_weight, expr_weights, cam_scales, cam_rots, cam_tvecs)
                contour = find_contour(np.array(lmk_2d_list[index])[self.contour['full_index']], pts2d)
                new_contour =  pts2d[contour]

                gt_lmk_img = vis.draw_pts(img, np.array(lmk_2d_list[index]), color=(0,0,255), width=1000, caption = "Ground Truth Landmark")
                pred_lmk_img = vis.draw_pts(img, pts2d[lmk_idx_list[index]], color=(0,0,255), width=1000, caption = "Fitting Landmark")
                pred_pts_img = vis.draw_pts(img, pts2d, color=(0,0,255), width = 1000, radius = 1, caption = "Fitting Landmark : iteration : "+iter_str)
                # new_cont_img = vis.draw_pts(img, new_contour, color=(0,0,255))
                new_cont_img = vis.draw_contour(img, pts2d, contour, color=(0,0,255), line_color=(0,255,0), caption=" ")
                gt_lmk_img = vis.draw_pts(new_cont_img, np.array(lmk_2d_list[index])[self.contour['full_index']], color=(0,255,255))
                mesh_contour_img = vis.draw_contour(gt_lmk_img, pts2d, self.mesh_boundary_index, color=(255,0,0), width =1000, caption = "Contour Landmark Selection based on Covexhull")
                vis.set_delay(1)


                mesh_overlay_image = vis.draw_mesh_to_img(img, Q_list[index], Rt_list[index], vv, self.f, (1.0, 0, 0), width=1000)
                

                #########################################################
                #draw inner shapes
                pred_lmk_img = vis.draw_pts_mapping(img, pts2d[np.array(lmk_idx_list[index])[inner_face_lmk_idx]],np.array(lmk_2d_list[index])[inner_face_lmk_idx], color=(255,0,0))
                gt_lmk_img = vis.draw_pts(pred_lmk_img, np.array(lmk_2d_list[index])[inner_face_lmk_idx], color=(0,255,255))
                pred_lmk_img = vis.draw_pts(gt_lmk_img, pts2d[np.array(lmk_idx_list[index])[inner_face_lmk_idx]],color=(0,0,255),width=1000, caption = "Mapping Landmark")
                # checking cors mapping 


                concat_img = vis.concatenate_img(2,2, pred_pts_img, mesh_contour_img, pred_lmk_img, mesh_overlay_image)
                vis.save(osp.join(root_path, name+"_{}".format(postfix)+".png"), concat_img)
                concat_img = vis.resize_img( concat_img, 1000)
                vis.show("test", concat_img )

                    

        iter_num = 3 
        # expr_weights = [np.zeros((expr_num, 1)) for _ in range(len(self.img_and_info.keys()))]
        expr_weights = [np.zeros((expr_num, 1)) for _ in range(len(self.img_list))]
        id_weight = np.zeros((id_num, 1))
        time_t = True
        Q_list = [[] for _ in range(len(self.img_list))]
        Rt_list = [[] for _ in range(len(self.img_list))]
        lmk_2d_list = [ info['lmk_data'] for info in self.img_list ]
        
        for iter_i in tqdm.tqdm(range(iter_num)):
            # expr_weights = [np.zeros((expr_num, 1)) for _ in range(len(self.img_list))]

            for key_id, item in tqdm.tqdm(enumerate(self.img_and_info.values())): 
                sel_imgs = [info['img_data'] for info in item ]
                lmk_2ds = np.array([info['lmk_data'] for info in item ])
                index_list =  [ info['index'] for info in item ]
                sel_lmk_idx_list = [ lmk_idx_list[info['index']] for info in item ]

                # nuetral, ids, exprs
                
                if iter_i == 0 : # if first iteration, we only compute with inner lmk.
                    # inner_sel_lmk_idx_list = [[lmk_idx_list[idx] for idx in inner_face_lmk_idx] for lmk_idx_list in sel_lmk_idx_list]
                    # sel_faces = [get_bars(neutral, ids, expr, idx_list) for idx_list in inner_sel_lmk_idx_list]
                    # raw_Q_Rt_list = [find_camera_matrix( get_combine_bar_model(face[0], face[1], face[2], id_weight, expr_weights[index]) ,lmk_2d[inner_face_lmk_idx], None) \
                    #                  for index, face, lmk_2d in zip(index_list, sel_faces, lmk_2ds) ]
                    sel_faces = [get_bars(neutral, ids, expr, idx_list) for idx_list in sel_lmk_idx_list]
                    raw_Q_Rt_list = [ find_camera_matrix( get_combine_bar_model(face[0], face[1], face[2], id_weight, expr_weights[index]) ,lmk_2d, None) \
                                     for index, face, lmk_2d in zip(index_list, sel_faces, lmk_2ds) ]
                    for i, ind in enumerate(index_list):
                        Q, Rt = raw_Q_Rt_list[i]
                        Q_list[ind] = Q
                        Rt_list[ind] = Rt
                else:
                    sel_faces = [get_bars(neutral, ids, expr, idx_list) for idx_list in sel_lmk_idx_list]
                    raw_Q_Rt_list = [ find_camera_matrix( get_combine_bar_model(face[0], face[1], face[2], id_weight, expr_weights[index]) ,lmk_2d, None) \
                                    for index, face, lmk_2d in zip(index_list, sel_faces, lmk_2ds) ]
                    for i, ind in enumerate(index_list):
                        Q, Rt = raw_Q_Rt_list[i]
                        Q_list[ind] = Q
                        Rt_list[ind] = Rt
                    # expr_Q_list = [ Q for Q, _ in raw_Q_Rt_list]
                    # expr_Rt_list = [ Rt for _, Rt in raw_Q_Rt_list ] 
                # else:
                    # sel_faces = [get_bars(neutral, ids, expr, idx_list) for idx_list in sel_lmk_idx_list]
                    # raw_Q_Rt_list = [ find_camera_matrix( get_combine_bar_model(face[0], face[1], face[2], id_weight, expr_weights[index]) ,lmk_2d, None) \
                                    #  for index, face, lmk_2d in zip(index_list, sel_faces, lmk_2ds) ]
                # expr_Q_list = [ Q for Q, _ in raw_Q_Rt_list]
                # expr_Rt_list = [ Rt for _, Rt in raw_Q_Rt_list ] 
                
                def exp_cost_builder(Q, neutral, ids, exps):
                    def wrapper(x,y):
                        w = x[-6:, 0]
                        Rt = get_Rt(*w.ravel())
                        weight_norm = w.T@w
                        id_weight = x[:len(ids), :]
                        expr_weight = x[len(ids):-6, :]

                        face_weight = x[6:, :]
                        face_weight_norm = face_weight.T @ face_weight

                        return default_cost_function(Q, Rt, neutral, ids, exps, id_weight, expr_weight, y )# + face_weight_norm 

                    return wrapper
                def exp_cost_builder2(Q, Rt, neutral, ids, exps):
                    def wrapper(x,y):
                        id_weight = x[:len(ids), :]
                        expr_weight = x[len(ids):, :]

                        face_weight = x
                        face_weight_norm = face_weight.T @ face_weight

                        return default_cost_function(Q, Rt, neutral, ids, exps, id_weight, expr_weight, y ) #+ face_weight_norm 

                    return wrapper
                

                def exp_cost_builder3(Q, Rt, neutral, ids, exps, id_weights):
                    def wrapper(x,y):
                        face_weight = x
                        # face_weight_norm = face_weight.T @ face_weight

                        return default_cost_function(Q, Rt, neutral, ids, exps, id_weights, x, y ) #+ face_weight_norm 

                    return wrapper
                
                def exp_cost_builder4(Q, neutral, ids, exps, id_weight):
                    def wrapper(x,y):
                        expr_weight = x[:-6, :]
                        w = x[-6:, :]
                        Rt = get_Rt(*w.ravel())

                        return default_cost_function(Q, Rt, neutral, ids, exps, id_weight, expr_weight, y )# + face_weight_norm 
                    return wrapper
                
                

                def exp_mapper(exps, index_mapper):
                    dims = len(exps.shape)
                    if dims == 3 :
                        return exps[index_mapper, ...]
                    elif dims == 2:
                        return exps[:, index_mapper]

                def weight_mapper(w, index_mapper):
                    return w[index_mapper, :]

                

                path_name = osp.join("cd_test", str(key_id))
                if not os.path.exists(path_name):
                    os.makedirs(path_name)
                save_prev = ""
                for image_i,( (neutral_, ids_, exprs_), lmk2d, sel_lmk_idx) in enumerate(zip(sel_faces, lmk_2ds, sel_lmk_idx_list)) : 
                    print("category :  " , image_i)
                    name = self.img_list[index_list[image_i]]['name']
                    
                    Rt = Rt_list[ index_list[image_i] ]
                    Q = Q_list[index_list[image_i] ]
                    # Q = expr_Q_list[image_i]
                    # Rt = expr_Rt_list[image_i]
                    # for con_i, idx in enumerate(self.contour['full_index']):
                    #     sel_lmk_idx[idx] = contour[con_i]
                    
                    neutral_, ids_, exprs_ = get_bars(neutral, ids, expr, sel_lmk_idx)
                    
        


                    init_weight = np.ones((len(exprs_) + len(ids_),1), dtype=np.float64)
                    rrr = np.random.uniform(0, 1, init_weight.shape)
                    init_weight*= rrr
                    init_weight*= 0
                    
                    init_weight = np.ones((len(exprs_) + len(ids_)+6,1), dtype=np.float64)
                    init_weight*= 0
                    r_x, r_y, r_z = decompose_Rt(Rt)
                    tx, ty, tz = Rt[:, -1]
                    init_weight[-6:, :] = np.array([r_x, r_y, r_z, tx, ty, tz]).reshape(-1,1)


                    init_weight[:len(ids_), :] = id_weight
                    # init_weight[len(ids_):-6, :] = expr_weights[index_list[image_i]]
                    init_weight[len(ids_):-6, :] = 0.0

                    # if i == 0 : # if first iteration
                        # opt_result = coordinate_descent(exp_cost_builder(Q, neutral_, ids_, exprs_), init_weight, lmk2d[inner_face_lmk_idx], 3)
                        # opt_result = coordinate_descent(exp_cost_builder2(Q, neutral_, ids_, exprs_), init_weight, lmk2d[inner_face_lmk_idx], 3)
                        # opt_result = grad_descent(exp_cost_builder2(Q, neutral_, ids_, exprs_), init_weight, lmk2d[inner_face_lmk_idx], 3)
                        # pass
                    # else: # add contour
                        # opt_result = coordinate_descent(exp_cost_builder2(Q, neutral_, ids_, exprs_), init_weight, lmk2d, 3)
                    
                    def clip_function(x):
                        nonlocal ids_, exprs_
                        x[:len(ids_),0] = np.clip(x[:len(ids_), 0], -1.0, 1.0)
                        # x[len(ids_):-6,0] = np.clip(x[len(ids_):-6, 0], 0.0, 1 )
                        x[len(ids_):-6,0] = np.clip(x[len(ids_):-6, 0], 0.0, None )
                        len_x = len(x)
                        xxxx = list(range(len_x - 6, len_x - 3))
                        if i in xxxx:
                            x[i] %= np.pi*2
                        return x
                    
                    def clip_function2(x):
                        nonlocal ids_, exprs_
                        x[:len(ids_),0] = np.clip(x[:len(ids_), 0], -1.0, 1.0)
                        x[len(ids_):,0] = np.clip(x[len(ids_):, 0], 0.0, 1.0)
                        return x
                    def clip_function3(x):
                        nonlocal ids_, exprs_
                        x = np.clip(x, 0.0, 1.0)
                        return x
                    def clip_function4(x):
                        nonlocal ids_, exprs_
                        x[:-6, :] = np.clip(x[:-6, :], 0.0, 1.0)
                        len_x = len(x)
                        xxxx = list(range(len_x - 6, len_x - 3))
                        if i in xxxx:
                            x[i] %= np.pi*2
                        return x

                    # https://dl.acm.org/doi/pdf/10.1145/2070781.2024196

                    coordinate_descent_iter = 100
                    fflag = 6
                    if fflag == 0 :
                        opt_result, grad_history, alpha_history = coordinate_descent(exp_cost_builder(Q, neutral_, ids_, exprs_), init_weight, lmk2d, coordinate_descent_iter, clip_func=clip_function)
                        cam_weight = opt_result[-6:, 0]
                        exp_weight = opt_result[len(ids_):-6, :]
                        Rt_list[ index_list[image_i] ] = get_Rt(*cam_weight.ravel())
                        expr_weights[index_list[image_i]] = exp_weight
                        save_prev = "Rt_id_expr_"
                        print(self.mouth_names)
                        base = ""
                        for e, w in zip(self.mouth_names, exp_weight[self.mouth_coeff_index, :].ravel()):
                            base += "{} : {} \n".format(e, w)
                        print(base)
                        for eee in range(len(grad_history)):
                            np.savetxt("cd_test/grad_his.txt",grad_history[eee], fmt="%.3f")
                            np.savetxt("cd_test/alpha.txt",alpha_history[eee], fmt="%.3f")
                    elif fflag == 6:
                        
                        opt_result, grad_history, alpha_history = coordinate_descent(exp_cost_builder(Q, neutral_, ids_, exp_mapper(exprs_, self.norm_base_expr_mesh_index)), init_weight, lmk2d, coordinate_descent_iter, clip_func=clip_function)
                        cam_weight = opt_result[-6:, 0]
                        exp_weight = opt_result[len(ids_):-6, :]
                        exp_weight = weight_mapper(exp_weight, self.norm_base_expr_mesh_index_revert)

                        Rt_list[ index_list[image_i] ] = get_Rt(*cam_weight.ravel())
                        expr_weights[index_list[image_i]] = exp_weight
                        save_prev = "Rt_id_expr_"
                        print(self.mouth_names)
                        base = ""
                        for e, w in zip(self.mouth_names, exp_weight[self.mouth_coeff_index, :].ravel()):
                            base += "{} : {} \n".format(e, w)
                        print(base)
                        for eee in range(len(grad_history)):
                            np.savetxt("cd_test/grad_his.txt",grad_history[eee], fmt="%.3f")
                            np.savetxt("cd_test/alpha.txt",alpha_history[eee], fmt="%.3f")

                    
                    elif fflag == 1 :
                        init_weight = init_weight[:-6]
                        opt_result = coordinate_descent(exp_cost_builder2(Q, Rt, neutral_, ids_, exprs_), init_weight, lmk2d, coordinate_descent_iter, clip_func=clip_function2)
                        exp_weight = opt_result[len(ids_):, :]
                        # id_weight = opt_result[:len(ids_), :]
                        # Rt_list[ index_list[image_i] ] =Rt
                        expr_weights[index_list[image_i]] = exp_weight
                        save_prev = "id_expr_only_"
                    elif fflag == 2:
                        init_weight = init_weight[len(ids):-6, :]
                        opt_result, _, _ = coordinate_descent(exp_cost_builder3(Q, Rt, neutral_, ids_, exprs_, id_weight), init_weight, lmk2d, coordinate_descent_iter, clip_func=clip_function3)
                        exp_weight = opt_result
                        # Rt_list[ index_list[image_i] ] =Rt 
                        expr_weights[index_list[image_i]] = exp_weight
                        save_prev = "expr_only_"
                    elif fflag == 3: #without opt
                        # Rt_list[ index_list[image_i] ] =Rt
                        save_prev = "Rt_only_"

                    elif fflag == 4 : #with gradient descent method (BFGS)
                        opt_result = bfgs_method(exp_cost_builder(Q, neutral_, ids_, exprs_), init_weight, lmk2d, 0)
                        exp_weight = opt_result[len(ids_):-6, :]
                        # id_weight = opt_result[:len(ids_), :]
                        cam_weight = opt_result[-6:, 0]

                        Rt_list[ index_list[image_i] ] = get_Rt(*cam_weight.ravel())
                        expr_weights[index_list[image_i]] = exp_weight                        
                        save_prev="grad_descent_"
                    elif fflag == 5: #expr and Rt opt
                        # fitting camera first.
                        init_weight = init_weight[len(ids_):, :]
                        opt_result, _, _ = coordinate_descent(exp_cost_builder4(Q, neutral_, ids_, exprs_, id_weight), init_weight, lmk2d, coordinate_descent_iter, clip_func=clip_function4)
                        # fit identity and expression
                        expr_weights[index_list[image_i]] = opt_result[:len(exprs_), :]
                        cam_weight = opt_result[len(exprs_):, :]
                        new_Rt = get_Rt(*cam_weight.ravel())
                        Rt_list[ index_list[image_i] ] = new_Rt
                        save_prev = "expr_Rt_"
                    save_prev = "citer_{}_{}".format(str(coordinate_descent_iter), save_prev)
                    
                    save_png(path_name, "{}_iter_{}".format(save_prev,iter_i), *[index_list[image_i]], iteration=str(iter_i))
                    
                    # Rt_list[ index_list[image_i] ] = get_Rt(*cam_weight.ravel())
                    # Rt_list[ index_list[image_i] ] =Rt
                    # Q_list[ index_list[image_i] ] = Q_list[image_i]

                    # # vv = get_combine_model(id_weight, exp_weight)
                    # vv = get_combine_model(id_weight, np.zeros_like(expr_weights[0]))
                    # igl.write_triangle_mesh(os.path.join(path_name, "expression_" +str(image_i)+ ".obj"), vv, self.neutral_mesh_f)
                    cv2.imwrite(osp.join(path_name, name+".png"), sel_imgs[image_i])

                # expr_weights[key_id] = exp_weight
            # #===========================================================================================
            #     # expr_weights.append(exp_weight)
                time_t = False

                # save_png(path_name, "{}_iter_{}".format(save_prev,i), *index_list, iteration=str(iter_i))
                    

                
            

            # phase 2
            def id_cost_funciton_builder(Q_list, Rt_list, v_idx_sel_index_list, expr_weight_list, lmk_2d_list):
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
                
            def clip_function4(x):
                nonlocal ids_, exprs_
                x = np.clip(x, -1.0, 1.0)
                return x
            init_id_weight = np.zeros_like(id_weight, dtype = np.float32)
            res_id_weight, _, _ = coordinate_descent(id_cost_funciton_builder(Q_list, Rt_list, lmk_idx_list, expr_weights, lmk_2d_list), init_id_weight, None, coordinate_descent_iter, clip_func=clip_function4)
            id_weight  = res_id_weight
            print("id expression :", id_weight.ravel())
            
            np.save(osp.join(self.save_root_dir, "Q_list_iter_{}".format(iter_i)), np.array(Q_list))
            np.save(osp.join(self.save_root_dir, "Rt_list_iter_{}".format(iter_i)), np.array(Rt_list))

            for Q_id,(Q, Rt) in enumerate(zip(Q_list, Rt_list)):
                name = self.img_list[Q_id]['name']
                np.savetxt("cd_test/Q_iter_{}_{}".format(iter_i,name), Q)
                np.savetxt("cd_test/Rt_iter_{}_{}".format(iter_i,name), Rt)
            
            for i, info in enumerate(self.img_list):

                lmk_2ds = info['lmk_data']
                img_index =  info['index']
                sel_lmk_idx = lmk_idx_list[info['index']]
                vv = get_combine_model(id_weight, expr_weights[img_index])
                pts2d = add_Rt_to_pts(Q_list[img_index], Rt_list[img_index], vv)
                contour = find_contour(np.array(lmk_2d_list[img_index])[self.contour['full_index']], pts2d)

                for con_i, idx in enumerate(self.contour['full_index']):
                    sel_lmk_idx[idx] = contour[con_i]



        gen_expression = np.zeros_like(expr_weights[0])
        user_identity_v = get_combine_model(id_weight, gen_expression)
        import igl
        igl.write_triangle_mesh("cd_test/gen_identity.obj", user_identity_v, self.neutral_mesh_f)
        

        self.identity_v = user_identity_v
        
        self.expressions_v = np.zeros((len(gen_expression), self.identity_v.shape[0], self.identity_v.shape[1]))
        for i in range(len(gen_expression)):
            gen_expression[i, 0 ] = 1.0
            user_epxr_v = get_combine_model(id_weight, gen_expression)
            self.expressions_v[i, ...] = user_epxr_v
            igl.write_triangle_mesh("cd_test/gen_expression_{}.obj".format(str(i)), user_epxr_v, self.neutral_mesh_f)
            gen_expression[i, 0 ] = 0.0


        self.id_weight = id_weight
        np.save("./cd_test/identity_weight.txt", self.id_weight)


    def extract_train_set_blendshapes(self):
        """
            this method use shape_fit method's actor(user)-specific blendshapes result.
            neutral pose : user-specific neutral pose
            blendshapes : user specific blendshapes
            ===========================================================================
            return
                weights that are extracted from user-specific blendshapes.
        """

        if not hasattr(self, "id_weight"):
            self.id_weight = np.load("./cd_test/identity_weight.txt.npy")
            self.unconcern_mesh_idx = np.load("./unconcerned_pts.npy")
            self.contour_pts_idx = np.load("./contour_pts.npy")

        global lmk_idx
        # define user-specific identity weight and expression.
        lmk_idx = np.array(lmk_idx)
        lmk_idx_list = np.stack([lmk_idx for _ in range(len(self.img_list))],axis=0)
        
        import re
        rex = re.compile(r"iter_(\d+).npy")

        def matcher(x):
            nonlocal rex 
            return int(rex.findall(x)[0])

        Q_list_paths = glob.glob(osp.join(self.save_root_dir, "Q_list_iter_*"))
        Rt_list_paths = glob.glob(osp.join(self.save_root_dir, "Rt_list_iter_*"))
        recent_Q_list_path = sorted(Q_list_paths, key=matcher)[-1]
        recent_Rt_list_path = sorted(Rt_list_paths, key=matcher)[-1]

        Q_list = np.load(recent_Q_list_path)
        Rt_list = np.load(recent_Rt_list_path)


        neutral = self.neutral_mesh_v

        ids_meshes = np.array(self.id_meshes)
        ids = ids_meshes - np.expand_dims(neutral, axis=0)
        id_num,_,_ = ids.shape


        expr_meshes = np.array(self.expr_meshes)
        expr = expr_meshes - np.expand_dims(neutral, axis=0)
        # expr_bar = expr[..., lmk_idx, :]
        expr_num, _,_ = expr.shape

        gen_zero_expression_weight = np.zeros((expr_num, 1), dtype=np.float32)


        user_specific_neutral = self.get_combine_model(neutral=neutral, ids=ids, expr=expr, w_i = self.id_weight, w_e= gen_zero_expression_weight)
        user_specific_expr = expr
        # user_specific_neutral_bar, _, user_specific_expr_bar = self.get_bars(neutral=user_specific_neutral, 
        #                                                                      ids = ids, exprs=user_specific_expr, 
        #                                                                      sel_lmk_idx = lmk_idx)
        
        def clip_function(x):
            x[:-6, :] = np.clip(x[:-6, :], a_min=0.0, a_max = 1.0)
            x[-6:-3, :] %= 2*np.pi
            return x 
        
        def expression_cost_funcion(Q, Rt, neutral_bar, exprs_bar, expr_weight, y, alpha_star):
            #alpha star is predefined face_mesh.
            # x := Q + Rt + expr weight

            blended_pose = self.get_combine_bar_model( neutral_bar=neutral_bar, ids_bar =  None, expr_bar = exprs_bar, w_i = None, w_e = expr_weight)
            
            gen = self.add_Rt_to_pts(Q, Rt, blended_pose)
            z = gen - y
            new_z = z.reshape(-1, 1)
            w_reg = 10
            weight_diff = (expr_weight.reshape(-1,1) -  alpha_star.reshape(-1,1))
            new_z = new_z.T @ new_z +  w_reg * weight_diff.T @ weight_diff

            return new_z
        
        def camera_posit_func_builder(Q, neutral_bar ,exprs_bar):
            def camera_posit_func(expr_weight, pts2d, is_First  = False):
                nonlocal neutral_bar, exprs_bar
                if is_First :
                    ind = [ii for ii in range(len(neutral_bar)) if ii not in self.contour['full_index']]
                # ind = [ii for ii in range(len(neutral_bar)) if ii in self.contour['full_index']]
                    neutral_bar = neutral_bar[ind, :]
                    exprs_bar = exprs_bar[:, ind, :]
                    pts2d = pts2d[ind, :]

                pts3d = self.get_combine_bar_model( neutral_bar=neutral_bar, ids_bar =  None, expr_bar = exprs_bar, w_i = None, w_e = expr_weight)

                # _, Rt = self.find_camera_matrix(pts3d, pts2d, guessed_projection_Qmat=Q )

                succ, rvec, tvec = cv2.solvePnP(pts3d, pts2d, cameraMatrix=Q, distCoeffs=np.zeros((4,1)))
                rot = cv2.Rodrigues(rvec)[0]
                rx,ry,rz = self.decompose_Rt(rot)
                # rx, ry, rz = self.decompose_Rt(Rt)
                # tvec = Rt[:, -1].reshape(-1,1)
                return rx,ry,rz, tvec[0, 0 ], tvec[1, 0], tvec[2, 0]
            def reset_cam_param(new_netural_bar, new_expr_bar):
                nonlocal neutral_bar, exprs_bar
                neutral_bar = new_netural_bar
                exprs_bar = new_expr_bar
            return camera_posit_func, reset_cam_param
        
        def dummpy_camera_posit_func_builder(A, B, C):
            def camera_posit_func(e, p):
                rx,ry,rz = self.decompose_Rt(Rt)
                tx,ty, tz = Rt[:, -1].ravel()
                return rx,ry,rz,tx,ty,tz
            return camera_posit_func

            

        def exp_mapper(exps, index_mapper):
            dims = len(exps.shape)
            if dims == 3 :
                return exps[index_mapper, ...]
            elif dims == 2:
                return exps[:, index_mapper]

        def weight_mapper(w, index_mapper):
            return w[index_mapper, :]

        def cost_function_builder(Q, neutral_bar, expr_bar, alpha_star):
            def wrapper(x, y):
                w_e = x[:-6, 0]
                Rt= self.get_Rt(*x[-6:,0].ravel())
                return expression_cost_funcion(Q, Rt, neutral_bar, expr_bar, w_e,y, alpha_star)
            def redefine_bars(new_neutral_bar, new_expr_bar):
                nonlocal neutral_bar, expr_bar
                neutral_bar = new_neutral_bar 
                expr_bar = new_expr_bar

            return wrapper, redefine_bars
        
        for key_id, item in tqdm.tqdm(enumerate(self.img_and_info.values())): 
            
            
            sel_imgs = [info['img_data'] for info in item ]
            lmk_2ds = np.array([info['lmk_data'] for info in item ])
            index_list =  [ info['index'] for info in item ]
            name_list =  [ info['name'] for info in item ]
            pre_def_weights =  [ info['predef_weight'] for info in item ]
            # category_predef_weight = np.load(osp.join("./predefined_face_weight", "disgust" + ".npy"))

            # pre_def_weights =  [ category_predef_weight for info in item ]


            sel_lmk_idx_list = [ lmk_idx_list[info['index']] for info in item ]
            sel_lmk_idx_list = [np.copy(lmk_idx)  for _ in range(len(item))]
            for name, sel_pts3d_idx, sel_img, idx, lmk2d, pre_def_w in zip(name_list, sel_lmk_idx_list, sel_imgs, index_list, lmk_2ds, pre_def_weights):
                Q = Q_list[idx]
                Rt = Rt_list[idx]

                user_specific_neutral_bar, _, user_specific_expr_bar = self.get_bars(neutral=user_specific_neutral, 
                                                                             ids = ids, exprs=user_specific_expr, 
                                                                             sel_lmk_idx = sel_pts3d_idx)
                init_weight = np.ones ((len(expr_meshes)+6, 1), dtype=np.float32) * 0.1
                r_x,r_y,r_z = self.decompose_Rt(Rt)
                tx, ty, tz = Rt[:, -1]
                # pre_d_w = weight_mapper(pre_def_w[1:, :], self.norm_base_expr_mesh_index)
                pre_d_w = pre_def_w[1:, :]
                init_weight[:-6, :] = pre_d_w
                init_weight[-6:, :] = np.array([r_x, r_y, r_z, tx, ty, tz]).reshape(-1,1)
                
                # camera_posit_func_builder(init_weight[-6:, :])
                # res, _, _ = self.coordinate_descent(cost_function_builder(Q, user_specific_neutral_bar, \
                #                                                         exp_mapper(user_specific_expr_bar, self.norm_base_expr_mesh_index), alpha_star = np.zeros_like(init_weight[:-6, :])), \
                #                                     init_weight, lmk2d, iter_nums = 100 ,clip_func = clip_function )
                import copy
                reg_alpha_star = copy.deepcopy(init_weight[:-6, :])
                # cost_f, reset_param_f = cost_function_builder(Q, user_specific_neutral_bar, exp_mapper(user_specific_expr_bar, self.norm_base_expr_mesh_index), alpha_star = reg_alpha_star)
                cost_f, reset_param_f = cost_function_builder(Q, user_specific_neutral_bar, user_specific_expr_bar, alpha_star = reg_alpha_star)
                camera_func, reset_cam_param_f = camera_posit_func_builder(Q, user_specific_neutral_bar, user_specific_expr_bar)
                
                def contour_remap(w):
                    nonlocal reset_param_f, reset_cam_param_f, lmk2d, Q, sel_pts3d_idx
                    expr_weight = w[:-6, :]
                    Rt = self.get_Rt(*w[-6:, :].ravel())

                    pose3d = self.get_combine_model(user_specific_neutral, None, user_specific_expr, None, expr_weight)
                    pose2d = self.add_Rt_to_pts(Q, Rt,  pose3d)
                    contour = self.find_contour(lmk2d[self.contour['full_index']], pose2d)


                    for con_i, idx in enumerate(self.contour['full_index']):
                        sel_pts3d_idx[idx] = contour[con_i]
                    
                    user_specific_neutral_bar, _, user_specific_expr_bar = self.get_bars(neutral=user_specific_neutral, 
                                                                             ids = ids, exprs=user_specific_expr, 
                                                                             sel_lmk_idx = sel_pts3d_idx)
                    reset_param_f(user_specific_neutral_bar, user_specific_expr_bar)
                    reset_cam_param_f(user_specific_neutral_bar, user_specific_expr_bar)


                    return sel_pts3d_idx
                # res, _, _ = self.coordinate_descent_LBFGS(cost_f,
                #                                     init_weight, lmk2d, iter_nums = 100 ,clip_func = clip_function, 
                #                                     camera_refinement_func= camera_posit_func_builder(Q, user_specific_neutral_bar, user_specific_expr_bar),
                #                                     contour_mapper_func= contour_remap )
                
                res, _, _ = self.coordinate_descent_LBFGS(cost_f,
                                                    init_weight, lmk2d, iter_nums = 100 ,clip_func = clip_function, 
                                                    camera_refinement_func= camera_func,
                                                    contour_mapper_func= contour_remap,
                                                    **{"verbose":True,
                                                            "img" : sel_img, \
                                                            "lmk_idx": sel_pts3d_idx, \
                                                            "contour_index" : self.contour['full_index'] ,\
                                                            "mesh": {"Q" : Q, "color" :(0.8,0,0), "neutral": user_specific_neutral,"exprs": user_specific_expr, "f" : self.f}, \
                                                            "x_info": {"pts":(255,0,0), "line":(255,255,0)},\
                                                            "y_info": {"pts":(0,0,255), "line":(0,255,255)},\
                                                            "width":1000}\
                                                    )
                
                # res, _, _ = self.coordinate_descent_LBFGS(cost_f,
                #                                     init_weight, lmk2d, iter_nums = 100 ,clip_func = clip_function, 
                #                                     camera_refinement_func= camera_func,
                #                                     contour_mapper_func= None)
                # res, _, _ = self.coordinate_descent_LBFGS(cost_f,
                #                                     init_weight, lmk2d, iter_nums = 100 ,clip_func = clip_function, 
                #                                     camera_refinement_func= dummpy_camera_posit_func_builder(Q, user_specific_neutral_bar, user_specific_expr_bar),
                #                                     contour_mapper_func= None)

                # expr_weight = init_weight[:-6, :]
                expr_weight = res[:-6, :]
                
                cam_weight = res[-6:, :]
                # cam_weight = init_weight[-6:, :]
                new_Rt = self.get_Rt(*cam_weight.ravel())
                t = (reg_alpha_star - expr_weight)
                tt = t.T@t
                # expr_weight = weight_mapper(expr_weight, self.norm_base_expr_mesh_index_revert)
                expr_weight = expr_weight
                reg_alpha_star =reg_alpha_star
                # expr_weight = pre_def_w[1:, :]
                # self.save_png(osp.join(self.save_root_dir, "test"), "test_info_{}".format(name), user_specific_neutral, None, user_specific_expr, None, expr_weight, Q, new_Rt, sel_img, lmk2d, 3, alpha=reg_alpha_star)
                self.save_png(osp.join(self.save_root_dir, "test"), "test_info_{}".format(name), user_specific_neutral, None, user_specific_expr, None, expr_weight, Q, new_Rt, sel_img, lmk2d, 3, alpha=reg_alpha_star)


    def set_save_root_directory(self, path):
        self.save_root_dir = path
        if not osp.exists(path):
            os.makedirs(path)
        



        

        


        
if __name__ == "__main__":

    

    parser = argparse.ArgumentParser(description='parser')

    parser.add_argument( '--dir', type=str, default="./cd_test" )
    args    = parser.parse_args()
    p = PreProp("landmark/meta.yaml", "prep_data")
    p.build()
    print(len(lmk_idx))
    p.set_save_root_directory("./cd_test")
    # p.simple_camera_calibration(p.images[0], p.lmks[0], p.meshes[0][0], lmk_idx)
    p.shape_fit(p.id_meshes, p.expr_meshes, lmk_idx)
    p.extract_train_set_blendshapes()
