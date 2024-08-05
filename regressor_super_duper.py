import numpy as np 

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import logging
logging.basicConfig(filename='reg_regressor_Rt_weak_logger.log',  level=logging.DEBUG)

import yaml 
import multiprocessing as mt 
import cv2 
import functools 

import tqdm
import time
import visualizer as vis
import scipy.spatial as sp 
import igl
import copy
import scipy
from itertools import repeat
import scipy.optimize as opt
import re 
import geo_func as geo 

# np.random.seed(8888)

def similarity_transform2(src, dest):
    """
        only rot and scale
        src to dest scale and rotation
        # scale, R, trans, Scale*R Trans, Rt, Inverse

    """


    src_mean = np.mean(src,axis=0, keepdims=True)
    dest_mean = np.mean(dest, axis= 0, keepdims=True)
    centered_src = src - src_mean
    centered_dest = dest - dest_mean
    trans_vec = dest_mean-src_mean
    U, s, Vh = np.linalg.svd(centered_src.T@ centered_dest)
    V = Vh.T
    det = np.linalg.det(V@U.T)
    sig = np.eye(3,3)
    sig[-1,-1] = det
    R = V@ sig @ U.T
    rotated_src = (R@ centered_src.T).T 
    rotated_src = rotated_src.reshape(-1,1)
    flat_centerd_dest = centered_dest.reshape(-1,1)
    # xtx = np.diag(rotated_src@ rotated_src.T)
    # xty = np.diag(centered_src@ centered_dest.T)
    xtx = rotated_src.T @ rotated_src
    xty = rotated_src.T@ flat_centerd_dest
    scale = xty/ xtx

    xtx2 = xtx.T@xtx
    xty2 = xtx.T@xty 

    # S = xty2 / xtx2
    # Scale = np.linalg.norm(Scale)
    # RR = Scale * R
    RR =  scale*R
    vv = np.eye(4,4)
    vv2 = np.eye(4,4)
    vv3 = np.eye(4,4)
    vv4 = np.eye(4,4)
    vv[:-1, -1] = -src_mean
    vv2[:3, :3] = RR
    vv3[:-1, -1] = dest_mean

    vv4[:3, :3] = R
    # scale, R, trans, Scale*R Trans, Rt, Inverse
    return scale, R, trans_vec, vv3@vv2@vv, vv3@vv4@vv, (-vv)@vv4.T@(-vv3)

import data_loader as dl
 
full_index, eye, contour,mouse, eyebrow, nose  = dl.load_ict_landmark("./ict_lmk_info.yaml")
lmk_without_contour_idx = list(set(full_index) - set(contour['full_index']))


if __name__ == '__main__':
    mt.freeze_support()

    mt_pool = mt.Pool(os.cpu_count())

regression_inner_contour_indices = np.load(os.path.join("./predefined_face_weight", "regression_contour.npy"))

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
_, f = igl.read_triangle_mesh("data/generic_neutral_mesh.obj")
def proj(Q, Rt, p):
    M = Rt [:3, :3]
    t = Rt [:, -1, np.newaxis]
    pts_2d_h = (Q @ (M @ p.T + t)).T
    pts_2d = pts_2d_h[:, :-1] / pts_2d_h[:, -1, np.newaxis]
    return pts_2d


def add_to_pts(M, v):
    """
    M : (3,4) or (4,4)
    v : (N, 3)
    """
    r, c = M.shape
    t = np.zeros((3,1))
    Rot = M[:3, :3]
    if c == 4:
        t = M[:, -1, None]
    new_v = v 
    return (Rot@new_v.T + t).T



class FirstLevelFern:
    def __init__(self, Q, K=300, F=5, beta = 250, P = 400, name = 0):
        
        self.K = K 
        self.F = F 
        self.Q = Q
        self.P = P
        self.beta = beta 
        self.ferns = [SecondLevelFern(Q, F, beta, P, name = iid) for iid in range(K)]
        self.name = name

    @staticmethod
    def generate_offset_d(P = 400):
        #only consider x, y
        loc = np.zeros((1,3))
        samples = np.random.uniform(-1, 1, size=(P, 3))
        samples[: , -1] = 0.0
        return samples


    
    @staticmethod
    def calc_offset_d2(P):
        res = np.zeros(shape=(P, 3))
        res[:, 0] = np.random.uniform(-1*0.9, 1*0.9)
        res[:, 1] = np.random.uniform(-1*0.9, 1*0.9)
        res[:, 2] = np.random.uniform(-1*0.9, 1*0.9)
        return res
        
    @staticmethod
    # def calc_offset_d(Q, Ms, samples, max_width, max_height, P):
    def calc_offset_d(self, Q, samples, P, w, h, kappa = 1.0):
        """
            pts : N x K
            samples : Vx3
        """
        # R = Ms[:3,:3]
        # t = Ms[:, -1, np.newaxis]
        
        if hasattr(self, "nearest_index") and hasattr(self, "disp"):
            return self.nearest_index, self.disp


        # see detail : https://www.microsoft.com/en-us/research/wp-content/uploads/2013/01/Face-Alignment-by-Explicit-Shape-Regression.pdf
        # full_index, eye, contour, mouse, eyebrow, nose  = dl.load_ict_landmark("./ict_lmk_info.yaml")
        # left_pupil = np.mean(samples[eye['left_eye']['full_index'], :]  , axis=0)
        # right_pupil = np.mean(samples[eye['right_eye']['full_index'], :], axis=0)
        # kappa = np.sqrt(np.sum((right_pupil - left_pupil)**2))*0.3  # 
        # kappa = 0.05 # For testing
        # kappa = 0.01 * kappa
        kappa = np.sqrt(np.sum((samples[0] - samples[1])**2))*0.3 * kappa  # 
        print(kappa)
        
        
        # samples2d = (Q@(R@samples.T+t)).T
        samples2d = (Q@(samples.T)).T
        samples2d = samples2d[:, :-1] / samples2d[:, -1, np.newaxis]
        max = np.max(samples, axis=0)
        min = np.min(samples, axis=0)
        # kdtree =sp.KDTree(samples - np.mean(samples, axis=0, keepdims=True))
        # kdtree =sp.KDTree(samples2d)
        # disp = np.random.uniform(min-1, max+1, (P*2, 3))

        disp = np.zeros((P,3), dtype=np.float32)
        idx_list = list(range(len(disp)))
        num_P = P
        disp = np.random.normal(0, kappa, size = (num_P, 3))
        # disp = np.random.uniform(-kappa, kappa, size = (num_P, 3))
        # while True :
        #     disp[idx_list, :] = np.random.uniform(-kappa, kappa, size = (num_P, 3))#TODO
        #     # flag, l = FirstLevelFern.test_outofbound_samples(Q, Ms, disp, w, h)
        #     flag, l = FirstLevelFern.test_valid_pts(disp)
        #     if flag: # if good random sample it is .
        #         break
        #     else :
        #         num_P = len(l)
        #         idx_list = l


        # # random select index
        # nearest_indextmp = list(range(len(samples)))
        # nearest_index = []
        # for i in range(P//len(samples)):
        #     nearest_index += nearest_indextmp
        # randomly_selected_index = list(np.random.randint(low=0, high=len(samples), size=(P%len(samples))))
        # nearest_index += randomly_selected_index


        nearest_index = np.random.randint(0, len(samples), size=P).tolist()

        # kdtree =sp.KDTree(samples)
        # disp = np.random.normal(np.mean( samples ,axis=0, keepdims=True), 2,size= (10000, 3))
        # disp[:, -1] = 1
        # _, nearest_index = kdtree.query(disp)
        # disp2d = (Q@(R@disp.T+t)).T
        # disp2d = (Q@(R@disp.T+t)).T
        # for i, nearest_idx in enumerate(nearest_index):
            # disp[i, :] += samples[nearest_idx, :] 
        
        # disp2d = (Q@(disp.T)).T
        # disp2d = disp2d[:, :-1] / disp[:, -1, np.newaxis]
        # _, nearest_index = kdtree.query(disp2d)
        # print(set(nearest_index))
        # print(disp2d)
        # print(samples2d)
        # import matplotlib.pyplot as plt 
        
        # plt.scatter(disp2d[:, 0], disp2d[:, 1], color='r')
        # plt.scatter(samples2d[:, 0], samples2d[:, 1], color='b')
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')

        # disp_test = np.copy(disp)
        # for i in range(len(disp_test)):
            # disp_test += samples[nearest_index[i]]
        # ax.scatter(disp_test[:, 0], disp_test[:, 1], disp_test[:, 2], marker="x")
        # ax.scatter(samples[:,0], samples[:,1], samples[:,2], marker="o")
        # plt.show()
        # resdisp = np.zeros_like(disp)
        
        # for i, nearest_idx in enumerate(nearest_index):
        #     disp[i, :] -= samples[nearest_idx, :] 
        self.nearest_index, self.disp = nearest_index, disp
        return nearest_index, disp
    
    @staticmethod
    def calc_appearance_vector(Image, Q, M, S_init_pose, P, disp, nearset_index):
        """
        app vector is N x 1
        
        return 
            App vector : N x 1
            pixel loc : P x 2
        """
        if len(Image.shape) >=3:
            h, w , _ = Image.shape
        else:
            h, w = Image.shape
        # nearset_index, disp = FirstLevelFern.calc_offset_d(Q, M, S_init_pose, w, h, P)
        # Vi := (intensity : ndarray  Nx2,  P_points Nx3)
        p = np.zeros_like(disp) # randomly select P
        p[...] = S_init_pose[nearset_index, :] + disp[...]
        # for ii, ni in enumerate(nearset_index):
        #     # p[ni, :] = S_init_pose[ni] + disp[ii]
        #     p[ii, :] = S_init_pose[ni] + disp[ii]


        
        if len(Image.shape) >= 3 :
            img_intensity = cv2.cvtColor(Image,  cv2.COLOR_BGR2GRAY)
        else : 
            img_intensity = Image
        # img_intensity = FirstLevelFern.convert_RGB_to_intensity(Image)
        # img_intensity = FirstLevelFern.convert_RGB_to_intensity(Image)
        
        pts_2d = proj(Q, M, p)
        pts_2d[pts_2d[:, 0] < 0, 0] = 0
        pts_2d[pts_2d[:, 0] >= w, 0 ] = w -1 
        pts_2d[pts_2d[:, 1] < 0, 1] = 0
        pts_2d[pts_2d[:, 1] >= h, 1] = h -1 

        loc = pts_2d.astype(np.uint)


        # TODO this is experiments.
        # convert opencv coordinate system 
        # ======> +X Axis
        # ||
        # ||
        # || 
        # V
        # +Y axis
        # y axis convert to open cv axis
        loc = geo.convert_to_cv_image_coord(loc, h)

        intensity_vectors = img_intensity.T[loc[:, 0].ravel(), loc[:, 1].ravel()] # N x 1
        intensity_vectors = intensity_vectors.reshape(-1,1)
        
        #################### comment below if you stop visualizer.
        # anchor_pts = proj(Q, M, S_init_pose)
        # # anchor_pts[anchor_pts[:, 0] < 0,0] = 0
        # # anchor_pts[anchor_pts[:, 0] >= w,0] = w -1 
        # # anchor_pts[anchor_pts[:, 1] < 0,1] = 0
        # # anchor_pts[anchor_pts[:, 1] >= h,1] = h -1 

        # vis.set_delay(1000)
        # # test for draw
        # # import visualizer

        # im = np.copy(Image)
        # im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        # data = geo.convert_to_cv_image_coord(proj(Q, M, S_init_pose),h)
        # im3 = vis.draw_circle(data, im, colors=(0,0,255),radius=1, thickness=2)
        
        # im3 = vis.draw_circle(loc ,im3, colors=(255,0,0), radius=1)
        # im3 = vis.draw_pts_mapping(im3, data[nearset_index, :] ,loc, (0,255,0))
        # # im3 = vis.draw_pts_mapping(im3, anchor_pts[nearset_index, :], pts_2d, (0,255,255), thickness=1)
        # im3 = vis.resize_img(im3, 1000)
        
        # vis.show('calc', im3)

        # im = np.copy(Image)
        # im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

        
        # x_min, y_min = np.min(anchor_pts, axis=0)
        # ax_min, ay_min = np.min(pts_2d, axis=0)
        # if ax_min < x_min:
        #     x_min = ax_min 
        # if ay_min < y_min : 
        #     y_min = ay_min 
        # x_max, y_max = np.max(anchor_pts, axis=0)
        # ax_max, ay_max = np.max(pts_2d, axis=0)
        # if x_max < ax_max:
        #     x_max = ax_max 
        # if y_max < ay_max : 
        #     y_max= ay_max
        
        # span = 10
        
        # im = im[int(y_min)-span:int(y_max)+span, int(x_min)-span:int(x_max)+span, : ]
        # h, w , _  = im.shape
        
        # desired_width = 800
        # ratio = desired_width / w 
        # new_h = h*ratio 
        # im = cv2.resize(im, [int(new_h), int(desired_width)])
        # anchor_pts = ratio*(anchor_pts - np.array([[x_min-span,y_min - span]]))
        # pts_2d_c =np.copy(pts_2d) 
        # pts_2d_c = ratio*(pts_2d_c - np.array([[x_min-span,y_min - span]]))
        

        # im = vis.draw_circle(anchor_pts, im, colors=(0,0,255), radius=2)
        # im1 = vis.draw_circle(pts_2d_c, im, colors=(255,0,0), radius=2)
        # im1 = vis.draw_pts_mapping(im1, anchor_pts[nearset_index, :], pts_2d_c, (0,255,255), thickness=2)


        # # im3 = vis.resize_img(im1, 800)
        # im = vis.concatenate_img(1,2, im1, im3)
        # im = im3
        # vis.show("MMMM", im)
        


        return intensity_vectors, loc, nearset_index, disp
    
    @staticmethod
    def calc_covariance(Vi1, Vi2):
        """
            
        """
        vi1_mean = np.mean(Vi1, axis= 0 )
        vi2_mean = np.mean(Vi2, axis= 0)
        centric_vi1 = Vi1 - vi1_mean
        centric_vi2 = Vi2 - vi2_mean
        return np.mean(np.multiply(centric_vi1, centric_vi2))
    
    @staticmethod
    def convert_RGB_to_intensity(img):
        # see also https://gamedev.stackexchange.com/questions/116132/how-does-one-calculate-color-intensity
        img[:, :,  :]
        cmax = np.max( img, axis = -1 ) # h,w, 1
        cmin = np.min( img, axis = -1 ) # h,w, 1
        delta = cmax - cmin # h,w,1
        saturation = delta/cmax
        return saturation
    
    @staticmethod
    def test_outofbound_samples(Q,Rt, v, w, h):
        """
            if test is passed, then return true. else False
        """
        if v.shape[-1] == 3:
            v_H = np.concatenate([v, np.ones((len(v), 1))], axis=-1)
        else:
            v_H = v
        s = (Q @ Rt @ v_H.T).T
        s_2d = s[:, :-1]/s[:, -1, None]
        idx = np.concatenate([np.where(s_2d < 0)[0], np.where(s_2d[:, 0] >= w)[0], np.where(s_2d[:, 1] >= h)[0]], axis=-1)
        # reflag = np.all(s_2d >= 0) and np.all(s_2d[:, 0] < w) and np.all(s_2d[:, 1] < h)
        return len(idx)==0, list(set(idx))
        
    @staticmethod
    def test_valid_pts(disps):
        """
            if test is passed, then return true. else False
        """
        norm = np.linalg.norm(disps, axis=-1)
        idx = np.concatenate([np.where(norm > 1.0)[0], np.where(norm < 10e-5)[0]], axis=-1)
        return len(idx) == 0 , list(set(idx))
        
        
            
        


    @staticmethod
    def convert_RGB_to_luminance(img):
        # see also https://stackoverflow.com/questions/56198778/what-is-the-efficient-way-to-calculate-human-eye-contrast-difference-for-rgb-val/56237498#56237498
        # BGR
        return img[:,:, 0] * 0.0722 + img[:,:, 0] * 0.7152 + img[:,:, 0] * 0.2125

    @staticmethod
    def calc_appearance_vector_wrapper_mt(index,self, images, M_list , mean_shape , current_shapes , disp, nearest_index, P):
            image, M, current_shape = images[index], M_list[index],  np.squeeze(current_shapes[index, ...])
            
            # scale, R, trasnlate , combine_Rt, _ = similarity_transform2(mean_shape, current_shape)
            scale, R, trasnlate , scale_Rt, combined_Rt, _ = similarity_transform2(mean_shape, current_shape)
            # combine_Rt = combine_Rt[:3, :]
            # current_shape = add_to_pts(scale*R, current_shape)
            # inverted_disp = add_to_pts(scale*R, disp)
            inverted_disp = add_to_pts(R, disp)
            # inverted_disp = disp # TODO for testing

            V_i, _, _, _  = FirstLevelFern.calc_appearance_vector(image, self.Q, M, current_shape, P, inverted_disp, nearest_index)
            # self.pixel_loc_list.append(pixel_loc)
            return V_i.reshape(-1)

    
    @staticmethod
    def pre_initialize(self, data_size, lmk_size):
        if(hasattr(self, "is_pre_initialized")):
            return
        
        for fern in self.ferns:
            fern.pre_initialize_(data_size, lmk_size)
        self.P2 = np.zeros((self.P, self.P))
        
        self.is_pre_initialized = True
        

    def train(self, regression_targets, image_list, current_shapes, Ss, S_index_list, M_list, mean_shape, Gt, kappa=1.0):
        lmk_size, dim = regression_targets[0].shape
        if len(image_list[0].shape ) == 3 :
            img_h, img_w, _ = image_list[0].shape
        else:
            img_h, img_w = image_list[0].shape
        target_num, lmk_size, dim = regression_targets.shape 

        
        

        N = len(current_shapes) # TODO disable it when real-trainig. 
        FirstLevelFern.pre_initialize(self, N, lmk_size)
        # N = 5 # for testing. remove it when you train model. this is toy model
    
        # self.V_list = np.zeros((N, self.P)) # N x P x 1
        self.pixel_loc_list = [] # N x P x 2
        self.nearest_index_list = [] # N x P x 1

        # self.nearset_index, self.disp = FirstLevelFern.calc_offset_d(self.Q, M, neutral_v, self.P, img_w, img_h)
        # self.nearset_index, self.disp = FirstLevelFern.calc_offset_d(self.Q, M, S_init_pose, self.P, img_w, img_h)
        # self.nearset_index, self.disp = FirstLevelFern.calc_offset_d(self, self.Q, mean_shape, self.P, img_w, img_h)
        print("kapa", kappa)
        self.nearset_index, self.disp = FirstLevelFern.calc_offset_d(self, self.Q, mean_shape, self.P, img_w, img_h, kappa=kappa)
        
        # ig = vis.draw_circle(pixel_loc, image, color=(255,0,0))
        # ig = vis.resize(ig, 1000)
        # vis.show("test", ig)
        global mt_pool

        mt_f = functools.partial(
            FirstLevelFern.calc_appearance_vector_wrapper_mt, 
            self=self, images=image_list, M_list =M_list, mean_shape = mean_shape, current_shapes = current_shapes, disp = self.disp, nearest_index=self.nearset_index, P = self.P
            )
        self.V_list = mt_pool.map(mt_f, range(N))
        # for i, (reg, cur) in enumerate(zip(regression_targets, current_shapes)):
        #     scale, R, trans_vec, scaled_Rt, combined_Rt, _ = similarity_transform2(cur, mean_shape)
        #     regression_targets[i,...] = add_to_pts(scale*R, reg)


        # for i in range(N): # 
        #     image, M, current_shape = image_list[i], M_list[i],  np.squeeze(current_shapes[i, ...])
            
        #     # inverted_disp = (np.linalg.inv(rot_inv_list[i]) @ self.disp.T).T
        #     scale, R, trasnlate , combine_Rt, _ = similarity_transform2(mean_shape, current_shape)
        #     combine_Rt = combine_Rt[:3, :]
        #     # Rt = TwoLevelBoostRegressor.similarity_transform(mean_shape, current_shape)
        #     # s = vis.draw_circle(proj(Q, np.eye(3,4), current_shape), image)
        #     # sa = vis.draw_circle(proj(Q, np.eye(3,4), mean_shape), image)
        #     # a = vis.draw_circle(proj(Q, np.eye(3,4), add_to_pts(combine_Rt, self.disp) ), image)
        #     # qq = vis.draw_circle(proj(Q, np.eye(3,4), add_to_pts(combine_Rt, mean_shape)), image)
        #     # s = vis.resize_img(s, 600)
        #     # sa = vis.resize_img(sa, 600)
        #     # qq = vis.resize_img(qq, 600)
        #     # a = vis.resize_img(a, 600)
        #     # aa = vis.concatenate_img(2, 2, s,a, sa, qq)
        #     # vis.show("test", aa)

            
        #     # inverted_disp = add_to_pts(scale*R, self.disp)
        #     inverted_disp = add_to_pts(R, self.disp)
        #     # inverted_disp = self.disp
            

        #     # V_i, pixel_loc, nearest_index, _  = FirstLevelFern.calc_appearance_vector(image,self.Q, M, S_init_pose, self.P, self.disp ,self.nearset_index)
        #     V_i, pixel_loc, nearest_index, _  = FirstLevelFern.calc_appearance_vector(image,self.Q, M, current_shape, self.P, inverted_disp ,self.nearset_index)
        #     # V_i, pixel_loc, nearest_index, _  = FirstLevelFern.calc_appearance_vector(image,self.Q, M, Ss[S_index_list[i]], self.P, inverted_disp ,self.nearset_index)
        #     # self.nearest_index_list.append(nearest_index)
        #     self.pixel_loc_list.append(pixel_loc)
        #     self.V_list.append(V_i)
        


        
        # intensity
        # P x N 
        # self.V = np.array(self.V_list, dtype=np.float32).reshape(N, self.P) # N x P x 1
        self.V = np.array(self.V_list, dtype=np.int32)
        self.pixel_location = np.array(self.pixel_loc_list)
        



            




        # self.P2 = np.zeros((self.P, self.P))
        #calc P2 
        for i in range(self.P):
            for j in range(i, self.P):
            # for j in range((i+1)):
                correlation = FirstLevelFern.calc_covariance(self.V[:, i], self.V[:, j])
                self.P2[i, j] = correlation
                self.P2[j, i] = correlation
            
        prediction = np.zeros_like(regression_targets)
        # for ki in tqdm.trange(self.K): #second level regression
        pbar = tqdm.tqdm(self.ferns, desc="train fern(err : inf)", leave=False)
        vis.set_delay(1)
        im0_o = cv2.cvtColor(np.copy(image_list[0]), cv2.COLOR_GRAY2RGB) 
        pp = proj(Q, M_list[0], Gt[0])
        ppp = proj(Q, M_list[0], current_shapes[0])
        pp = geo.convert_to_cv_image_coord(pp, img_h)
        ppp = geo.convert_to_cv_image_coord(ppp, img_h)
        
        im0 = vis.draw_circle( pp, im0_o, (255,0,0), radius=2 )
        im0 = vis.draw_circle( ppp, im0, (0,0,255), radius=1 )
        im0 = vis.resize_img( im0, 600 )
        vis.put_text(im0, "origin 0", color=(0,0,255))
        im1_o = cv2.cvtColor(np.copy(image_list[-1]), cv2.COLOR_GRAY2RGB) 
        
        pp = proj(Q, M_list[-1], Gt[-1])
        ppp= proj(Q, M_list[-1], current_shapes[-1])
        pp = geo.convert_to_cv_image_coord(pp, img_h)
        ppp = geo.convert_to_cv_image_coord(ppp, img_h)
        im1 = vis.draw_circle( pp, im1_o, (255,0,0), radius=2 )
        im1 = vis.draw_circle( ppp , im1, (0,0,255), radius=1 )
        im1 = vis.resize_img( im1, 600 )
        vis.put_text(im1, "origin -1", color=(0,0,255))
        ii = 0 
        pred = np.zeros_like(regression_targets)
        for fern_regressor in pbar:

            fern_regressor.train(regression_targets, self.P2, self.V, self.pixel_location, self.nearset_index, pred)
            prediction += pred
            regression_targets -= pred
            pred[...] = 0

            pp = proj(Q, M_list[0], current_shapes[0] +  prediction[0])
            pp = geo.convert_to_cv_image_coord(pp, img_h)

            ppp = proj(Q, M_list[0], Gt[0])
            ppp = geo.convert_to_cv_image_coord(ppp, img_h)
            im2 = vis.draw_circle( ppp, im0_o, (255,0,0) , radius=2)
            im2 = vis.draw_circle( pp, im2, (0,0,255) , radius=1)
            im2 = vis.resize_img( im2, 600 )
            vis.put_text(im2, "pred 0", color=(0,0,255))

            pp = proj(Q, M_list[-1], current_shapes[-1] + prediction[-1])
            pp = geo.convert_to_cv_image_coord(pp, img_h)
            ppp = proj(Q, M_list[-1], Gt[-1])
            ppp = geo.convert_to_cv_image_coord(ppp, img_h)
            im3 = vis.draw_circle( ppp , im1_o, (255,0,0) ,radius=2)
            im3 = vis.draw_circle( pp, im3, (0,0,255) ,radius=1)
            im3 = vis.resize_img( im3, 600 )
            vis.put_text(im3, "pred -1 : {}".format(ii), color=(0,0,255))
            im = vis.concatenate_img(2, 2, im0, im1,im2, im3)
            vis.show("weak!!", im)
            ii+=1
            logging.debug('fern regressor : {} '.format(np.sum(np.sqrt(np.sum(regression_targets**2, -1)))))
            
            pbar.set_description("train fern(err : {})".format(np.sum(np.sqrt(np.sum(regression_targets**2, -1))) ))

        return prediction
            

    def predict(self, img, cur_S, S_prime, mean_shape, predRt, Q):
        res = np.zeros_like(cur_S)
        # R = predRt[:,:3]
        # RR = TwoLevelBoostRegressor.normalize_matrix(mean_shape, cur_S)

        # scale, R, trans_vec, combined_Rt, _, _ = similarity_transform2(mean_shape, cur_S)
        scale2, R, trans_vec, combined_Rt, _, _ = similarity_transform2(mean_shape, cur_S)

        # Rtinv = TwoLevelBoostRegressor.inverse_Rt(predRt)
        Rtinv = predRt
        # vis.draw_circle(proj(Q, Rtinv, p))
        # inverted_disp = add_to_pts(scale2*R,self.disp)
        inverted_disp = add_to_pts(R,self.disp)
        # meanspace_cur_S = add_to_pts(scale*R, cur_S)
        # inverted_disp = self.disp
        i1 = vis.draw_circle(geo.convert_to_cv_image_coord(proj(Q, Rtinv, cur_S[self.nearest_index_list, :] + inverted_disp), 1080), img, radius=1)
        # i2 = vis.draw_circle(proj(Q, combined_Rt, mean_shape), img)
        # vis.show("smalll", vis.resize_img(i1,1000))
        
        # intensity_vectors, loc, nearset_index, disp  = FirstLevelFern.calc_appearance_vector(img, Q, Rtinv, cur_S, 400, inverted_disp, self.nearest_index_list)
        # im = vis.draw_circle(proj(Q, np.eye(3,4), cur_S),img, colors=(0,0,255))
        i = 0 
        # intensity_vectors, loc, nearset_index, disp  = FirstLevelFern.calc_appearance_vector(img, Q, Rtinv, cur_S, 400, inverted_disp, self.nearest_index_list)
        intensity_vectors, loc, nearset_index, disp  = FirstLevelFern.calc_appearance_vector(img, Q, Rtinv, cur_S, 400, inverted_disp, self.nearest_index_list)
        pred = np.zeros_like(cur_S)
        for fern_reg in self.ferns:
            pred += fern_reg.pred(img, cur_S, Q, predRt, intensity_vectors)
            # pred = fern_reg.pred(img, cur_S, Q, Rtinv, intensity_vectors, inverted_disp, self.nearest_index_list)
            # cur_S += pred
            # res += pred
            # if i % 100 == 0 :
            #     im = vis.draw_circle(proj(Q, Rtinv, cur_S),img, colors=(0,0,255), radius=1)
            #     im = vis.resize_img(im,1000)
            #     vis.put_text(im, str(i), color=(255,0,0))
            #     vis.show("small", im)
            # i+=1
        # res = add_to_pts(R.T, res)
        # scale, R, trans_vec, combined_Rt, _, _ = similarity_transform2(mean_shape, cur_S)

        # pred = add_to_pts(scale*R, pred)
        return pred

    def load(self, root_path):
        directory = "weak_regressor_"+str(self.name)
        save_path = os.path.join(root_path, directory)

        self.V = np.load(os.path.join(save_path, "V.npy"))
        self.P2 = np.load(os.path.join(save_path, "P2.npy"))
        self.disp = np.load(os.path.join(save_path, "disp.npy"))
        self.pixel_location = np.load(os.path.join(save_path, "pixel_location.npy"))
        self.nearest_index_list = np.load(os.path.join(save_path, "nearest_index.npy"))
        for f in self.ferns:
            f.load(save_path)

    def save(self, root_path):
        directory = "weak_regressor_"+str(self.name)
        save_path = os.path.join(root_path, directory)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, "V"), self.V)
        np.save(os.path.join(save_path, "P2"), self.P2)
        np.save(os.path.join(save_path, "disp"), self.disp)
        np.save(os.path.join(save_path, "pixel_location"), self.pixel_location)
        np.save(os.path.join(save_path, "nearest_index"), self.nearset_index)
        for fern in tqdm.tqdm(self.ferns):
            fern.save(save_path)


class SecondLevelFern:
    def __init__(self, Q, F=5, beta = 250, P = 400, name=0):
        self.F = F 
        self.Q = Q
        self.P = P
        self.beta = beta 
        self.name = name
    # def pred(self, img, cur_S, Q,Rt, intensity_vector, disp, nearest_index):
    def pred(self, img, cur_S, Q,Rt, intensity_vector):
        index = 0
        # debug_index = np.zeros(self.F, dtype=np.int32)
        self.selected_nearest_index = self.selected_nearest_index.astype(np.uint)
        for f in ( range( self.F)):
            nearest_lmk_idx_1 = self.selected_nearest_index[f, 0]
            nearest_lmk_idx_2 = self.selected_nearest_index[f, 1]
            # x = self.selected_pixel_location[f, 0]
            # y = self.selected_pixel_location[f, 1]
            nearest_lmk_idx_1 = self.selected_pixel_index[f, 0]
            nearest_lmk_idx_2 = self.selected_pixel_index[f, 1]
            
            
            sps = img.shape
            if(len(sps) == 3):
                h, w, _ = img.shape
            else:
                h, w = img.shape


            # shape1 = cur_S[nearest_lmk_idx_1, None]
            # proj_xy = proj(Q, Rt, shape1)
            # proj_xy[proj_xy < 0] = 0
            # proj_xy[proj_xy[:, 0] >= w, 0] = w-1
            # proj_xy[proj_xy[:, 1] >= h, 1] = h-1
            # x,y = proj_xy.astype(np.uint).ravel()
            # intensity_1 = img[y,x].astype(np.float32)
            intensity_1 = intensity_vector[nearest_lmk_idx_1]
            intensity_2 = intensity_vector[nearest_lmk_idx_2]
            
            # shape2 = cur_S[nearest_lmk_idx_2, None]
            # proj_xy = proj(Q, Rt, shape2)
            # proj_xy[proj_xy < 0] = 0
            # proj_xy[proj_xy[:, 0] >= w, 0] = w-1
            # proj_xy[proj_xy[:, 1] >= h, 1] = h-1
            # x,y = proj_xy.astype(np.uint).ravel()
            # intensity_2 = img[y,x].astype(np.float32)

            # intensity_1 = intensity_vector[nearest_lmk_idx_1]
            # intensity_2 = intensity_vector[nearest_lmk_idx_2]


            # S_H = (Q @ cur_S.T).T
            # S = S_H[:, :-1] / S_H[:, -1, None]
            # project_xy = S + self.selected_pixel_location[f , :2]
            # project_xy = project_xy[int(nearest_lmk_idx_1), :]
            # h,w = img.shape
            # min_pj_xy =  np.minimum(project_xy, np.expand_dims([w - 1,h - 1],0))
            # xy = np.maximum([[0,0]],min_pj_xy)
            # xy = xy.astype(np.uint)
            # intensity_1 = img [xy[:,1].ravel(), xy[:, 0].ravel()]

            # project_xy = S + self.selected_pixel_location[f , 2:]
            # project_xy = project_xy[int(nearest_lmk_idx_2), :]
            # xy = np.maximum([[0,0]], np.minimum(project_xy, np.expand_dims([w - 1,h - 1],0)))
            # xy = xy.astype(np.uint)
            # intensity_2 = img [xy[:, 1].ravel(), xy[:, 0].ravel()]
            diffs = intensity_1.astype(np.float32) - intensity_2.astype(np.float32)
            if(diffs >= self.Fern_threshold[f]):
                index = index +int(pow(2, f))
                # debug_index[f] = 1
        # print(self.name, " : ", debug_index)
        return self.bin_params[index]



    @staticmethod
    def _precompute_corr_(cor_index, corr_mat, P2, sample_y_variances):
        res = []
        for i in range(len(P2)):
                for j in range(i+1):
                # for j in range(len(P2)):
                    index_pair = (i, j)
                    inverse_index_pair = (j, i)

                    temp = (P2[i,i] + P2[j, j] - 2*P2[i, j])*sample_y_variances[cor_index]
                    if abs(temp) < 1e-10:
                        continue                    
                    corr = (corr_mat[cor_index, i] - corr_mat[cor_index, j])/np.sqrt(abs(temp))
                    permuted_index_corr = -corr
                    selected_corr = corr if corr >= permuted_index_corr else permuted_index_corr
                    selected_index_pair = index_pair if corr >= permuted_index_corr else inverse_index_pair
                    res.append((selected_index_pair, selected_corr))

        res.sort(key=lambda x : x[-1], reverse=True)
        return res


        
    @staticmethod
    def _classify_train_data_mt(v_i, self, V):
        index = 0
        for j in range(self.F):
            index_m = self.selected_pixel_index[j, 0]
            index_n = self.selected_pixel_index[j, 1]
            density1 = V[v_i, index_m]
            density2 = V[v_i, index_n]
            pixel_diff = density1 - density2 
            if pixel_diff >= self.Fern_threshold[j]:
                index = int(index + pow(2.0,j))
        
        return v_i, index
    
    def pre_initialize_(self, data_size, lmk_size):
        if(hasattr(self, "is_pre_initialized")):
            return 
        N = data_size
        self.selected_pixel_index = np.zeros((self.F, 2), dtype=np.uint)
        self.selected_nearest_index = np.zeros((self.F, 2))
        self.selected_pixel_location = np.zeros((self.F, 4), dtype=np.uint)
        self.ci_mat = np.zeros((self.F,N), dtype=np.float32)
        self.corr_mat = np.zeros((self.F, self.P))
        self.Fern_threshold = np.random.uniform(0, 1, (self.F, 1))
        bin_size = 2**self.F
        self.bin_params = np.zeros((bin_size, lmk_size, 3), dtype=np.float32)
        
        self.zero_mat = np.zeros((lmk_size, 3))
        self.data_in_bins = [[] for i in range(bin_size)]
        
        self.is_pre_initialized = True 


    def train(self, regression_targets_S,P2, V, pixel_location, nearest_indices, prediction):
        N = len(regression_targets_S) 
        # N = 5# TODO for test
        num_targets, lmk_size, dim = regression_targets_S.shape
        
        self.pre_initialize_(N, lmk_size)
        
        # self.selected_pixel_index = np.zeros((self.F, 2), dtype=np.uint)
        # self.selected_nearest_index = np.zeros((self.F, 2))
        # self.selected_pixel_location = np.zeros((self.F, 4), dtype=np.uint)
        # self.ci_mat = np.zeros((self.F,N), dtype=np.float32)
        
        # self.corr_mat = np.zeros((self.F, self.P))
        # self.corr_mat = np.zeros((self.F, self.P))

        # self.Fern_threshold = np.random.uniform(0, 1, (self.F, 1))
        
        # start_time = time.time()






        for fi in range(self.F):
            # direction
            # Y_fi = np.random.normal(0, 1, size=(lmk_size*dim)) # same as lmk_num, 2
            Y_fi = np.random.normal(0, 1, size=(lmk_size,dim)) # same as lmk_num, 2
            Y_fi_norm = np.linalg.norm(Y_fi, axis=-1)[..., np.newaxis]
            Y_fi /= Y_fi_norm 

            Y_fi = Y_fi.reshape(-1,1)
            y_proj = regression_targets_S.reshape(N, -1) @ Y_fi
            self.ci_mat[fi, ...] = y_proj.reshape(1,-1)
        sample_y_variances = np.var(self.ci_mat, axis=-1)
        # sample_y_variances = np.ones ((len(self.ci_mat), 1), dtype=np.float32)
        #(F, N)
        # calc covariacne between y_proj, P2
        for Fi in range(self.F):
            for Pj in range(self.P):
                self.corr_mat[Fi, Pj] = FirstLevelFern.calc_covariance(self.ci_mat[Fi, :], V[:, Pj])

        # corr_nominator = np.zeros((self.F, self.P, self.P))
        # corr_nominator.ctypes()
        # corr_nominator_ptr = np.ctypeslib.as_ctypes(corr_nominator)
        # P2
        # self.corr_mat
        # mt_pool.map
        

        #precompute
        precompute_corr_f = functools.partial(SecondLevelFern._precompute_corr_, corr_mat = self.corr_mat, P2 = P2, sample_y_variances=sample_y_variances)
        # precomputed_corr = mt_pool.starmap(SecondLevelFern._precompute_corr_, zip(range(self.F), repeat(self.corr_mat), repeat(P2), repeat(sample_y_variances)))
        precomputed_corr = mt_pool.map(precompute_corr_f, range(self.F))
        
        
        def check_is_same(a, barray, lim_size=None):
            res_flag = False 
            barray = barray[:lim_size]
            for b in barray:
                b_n = (b[0], b[1])
                b_r = (b[1], b[0])
                for bb in [b_n, b_r]:
                    res_flag |= a == bb
            return res_flag

                            

        for mpi in range(self.F):
            index_corr_pairs = precomputed_corr[mpi]
            max_pixel_ind = (0,0)
            for index_pair, corr_val in index_corr_pairs:
                if(check_is_same(index_pair, self.selected_pixel_index, mpi+1)):
                    continue 
                else : 
                    max_corr = corr_val
                    max_pixel_ind = index_pair
                    break

            self.selected_pixel_index[mpi, :] = max_pixel_ind
            # self.selected_pixel_location[mpi, :2] = pixel_location[max_pixel_ind[0]]
            # self.selected_pixel_location[mpi, 2:] = pixel_location[max_pixel_ind[1]]
            self.selected_nearest_index[mpi, 0] =  nearest_indices[max_pixel_ind[0]]
            self.selected_nearest_index[mpi, 1] =  nearest_indices[max_pixel_ind[1]]
                    
        # aaaaas=[]
        # aaa = np.copy(self.selected_pixel_index)
        # for mpi in range(self.F):
        #     max_corr =  -1
        #     max_pixel_ind = (0,0)
        #     for mpj in range(self.P):
        #         for mpk in range(self.P):
        #         # for mpk in range(0, mpj+1):
                    
        #             temp = (P2[mpj, mpj] + P2[mpk, mpk] - 2*P2[mpj, mpk])*sample_y_variances[mpi]
        #             if abs(temp) < 1e-10:
        #                 continue
                    
        #             flag = False
        #             for pi in range(mpi):
        #                 if mpj == self.selected_pixel_index[pi, 0] \
        #                     and mpk == self.selected_pixel_index[pi, 1]:
                            
        #                     flag = True
        #                     break
        #                 elif mpj == self.selected_pixel_index[pi, 1] \
        #                     and mpk == self.selected_pixel_index[pi, 0]:
                            
        #                     flag = True 
        #                     break
                
        #             if flag:
        #                 continue
                    
        #             corr = (self.corr_mat[mpi, mpj] - self.corr_mat[mpi, mpk]) / np.sqrt(abs(temp))
        #             if corr > max_corr:
        #                 max_corr = corr
        #                 max_pixel_ind = (mpj, mpk)
            
        #     aaaaas.append(max_corr)
        #     # self.selected_pixel_index[mpi, :] = max_pixel_ind
        #     aaa[mpi, :] = max_pixel_ind
        #     # self.selected_pixel_location[mpi, :2] = pixel_location[max_pixel_ind[0]]
        #     # self.selected_pixel_location[mpi, 2:] = pixel_location[max_pixel_ind[1]]
        #     self.selected_nearest_index[mpi, 0] =  nearest_indices[max_pixel_ind[0]]
        #     self.selected_nearest_index[mpi, 1] =  nearest_indices[max_pixel_ind[1]]
        
        # max_diff = -1 # make threshold
        for ii in range(self.F):
            index_m = self.selected_pixel_index[ii, 0]
            index_n = self.selected_pixel_index[ii, 1]
            pixel_diffs = abs(V[:, index_m] - V[:, index_n]) # N x 1
            max_diff_idx = np.argmax(pixel_diffs)
            max_diff = pixel_diffs[max_diff_idx]

            # for intensity_j, intensity_k in zip(V[:, index_m], V[:, index_n]) : 
                # temp = intensity_j - intensity_k
                # if abs(temp) > max_diff:
                    # max_diff = abs(temp)
            self.Fern_threshold[ii] = np.random.uniform(-0.2*max_diff, 0.2*max_diff) 
        
        # end_time = time.time()
        # print("spent time 1 : %s" %(end_time - start_time))
        # start_time = end_time 
        
        bin_size = 2**self.F
        # self.data_in_bins = [[] for i in range(bin_size)]
        classify_wrapper_mt = functools.partial(SecondLevelFern._classify_train_data_mt, self=self, V=V)
        Vi2bin_index_pair_result = mt_pool.map(classify_wrapper_mt, range(N))
        for v_i, bin_index in Vi2bin_index_pair_result:
            self.data_in_bins[bin_index].append(v_i)
        # for i in range(N):
        #     index = 0
        #     for j in range(self.F):
        #         index_m = self.selected_pixel_index[j, 0]
        #         index_n = self.selected_pixel_index[j, 1]
        #         density1 = V[i, index_m]
        #         density2 = V[i, index_n]
        #         pixel_diff = density1 - density2 
        #         if pixel_diff >= self.Fern_threshold[j]:
        #             index = int(index + pow(2.0,j))
        #     self.data_in_bins[index].append( i)

        # prediction for training. yeah-ah
        # end_time = time.time()
        # print("spent time 3 : %s" %(end_time - start_time))
        # start_time = end_time 
        
        # prediction = np.zeros_like(regression_targets_S)
        # self.bin_params = np.zeros((bin_size, lmk_size, 3), dtype=np.float32)
        # zero_mat = np.zeros((lmk_size, 3))
        for fi in range(bin_size):
            sel_bin_size = len(self.data_in_bins[fi])
            # for bi in range(sel_bin_size):
            #     # Si_list, S_i_init_list = data_in_bins[fi][bi] # N x T x 3, N x T x 3
            #     index = self.data_in_bins[fi][bi] # N x T x 3, N x T x 3
            #     temp += regression_targets_S[index]
                
            if (sel_bin_size == 0 ):
                    self.bin_params[fi] = self.zero_mat
                    continue 
            else : 
                temp = np.sum(regression_targets_S[self.data_in_bins[fi]], axis= 0)

            den = 1 + (self.beta/sel_bin_size)
            no = 1
            coeff = no/den
            delta  = (temp / sel_bin_size)
            delta_Sb =  coeff * delta
            # delta_Sb = (1/(1+0 / sel_bin_size)) * (temp / sel_bin_size)
            self.bin_params[fi] = delta_Sb
            # for bi in range(sel_bin_size):
            #     index = self.data_in_bins[fi][bi]
            #     prediction[index] += delta_Sb
            # prediction[self.data_in_bins[fi][bi], :] += delta_Sb[None, ...]
            prediction[self.data_in_bins[fi], ...] = delta_Sb[None, ...]
            
        
        return prediction

    def load(self, root_path):
        
        directory = "fern_regressor_"+str(self.name)
        save_path = os.path.join(root_path, directory)
        self.bin_params = np.load(os.path.join(save_path, "bin_param.npy") )
        self.data_in_bins = np.load(os.path.join(save_path, "data_in_bins.npy"), allow_pickle=True )
        self.Fern_threshold = np.load(os.path.join(save_path, "Fern_threshold.npy") )
        self.selected_pixel_index = np.load(os.path.join(save_path, "selected_pixel_index.npy") )
        self.selected_pixel_location = np.load(os.path.join(save_path, "selected_pixel_location.npy") )
        self.selected_nearest_index = np.load(os.path.join(save_path, "selected_nearest_index.npy") )
        self.corr_mat = np.load(os.path.join(save_path, "corr_mat.npy") )
        self.ci_mat = np.load(os.path.join(save_path, "ci_mat.npy") )
 

    def save(self, root_path):
        
        directory = "fern_regressor_"+str(self.name)
        save_path = os.path.join(root_path, directory)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, "bin_param") ,self.bin_params)
        # np.save(os.path.join(save_path, "data_in_bins") ,self.data_in_bins)
        np.save(os.path.join(save_path, "Fern_threshold") ,self.Fern_threshold)
        np.save(os.path.join(save_path, "selected_pixel_index") ,self.selected_pixel_index)
        np.save(os.path.join(save_path, "selected_pixel_location") ,self.selected_pixel_location)
        np.save(os.path.join(save_path, "selected_nearest_index") ,self.selected_nearest_index)
        np.save(os.path.join(save_path, "corr_mat") ,self.corr_mat)
        np.save(os.path.join(save_path, "ci_mat") ,self.ci_mat)


class TwoLevelBoostRegressor:

    """

    """


    def __init__(self, Q = None, neutral_v = None, Ss=None, S_original = None , T=10, K=300, F=5, beta = 250, P = 400, data=None):
    # def __init__(self, Q = None, T=1, K=20, F=1, beta = 250, P = 400):
        """
            T : The number of first level regression
            K : The number of second level regression
            F : size of bin in weak regressor 
            Q : camera intrinsic matrix (3x3)

        """
        self.nuetral_v = neutral_v
        self.T = T 
        self.K = K 
        self.F = F 
        self.Q = Q
        self.P = P
        self.beta = beta
        self.Ss = Ss
        self.S_original = S_original
        self.test_data = data
        
        
    def set_save_path(self, path):
        self.save_path = path 


    def save_model(self):
        if(not os.path.exists(self.save_path)):
            os.makedirs(self.save_path)

        np.save(os.path.join(self.save_path, "Q"), self.Q)
        np.save(os.path.join(self.save_path, "S"), self.S_list)
        meta_dict = {"T" : self.T, "K" : self.K, "F" : self.F, "Q" : "Q.npy", "P" : self.P, "beta" : self.beta, "lmk_size": self.lmk_size, "mean_shape" : "mean_shape.npy","S" : "S.npy"}
        with open(os.path.join(self.save_path,"meta.txt"), "w") as fp :
            yaml.dump(meta_dict, fp)
        for weak_reg in tqdm.tqdm(self.weak_regressors, "save weak regressors"):
            weak_reg.save(self.save_path)


    def detect_by_dlib(self, img, **kwargs):
        lmk = [[-1,-1] for _ in range(68)]
        if len(img.shape) == 3 :
            h,w, _ = img.shape
        else:
            h,w = img.shape

        predef_width = 640
        max_length_ratio = 1.0
        if h > 1000 or w > 1000:
            max_length_ratio = predef_width/w
        new_img = cv2.resize(img, [int(max_length_ratio*w),int(max_length_ratio*h)])
        img = new_img 
        rects = self.detector(img, 1)
        for i, rect in enumerate(rects):
            l = rect.left()
            t = rect.top()
            b = rect.bottom()
            r = rect.right()
            shape = self.predictor(img, rect)
            for j in range(68):
                x, y = shape.part(j).x, shape.part(j).y
                lmk[j][0] = x/max_length_ratio
                lmk[j][1] = y/max_length_ratio
        return np.array(lmk, dtype=np.float32)
    
    def refine_3d_lmk(self, lmk2d, shape_lmk_idx=None, **kwargs):
        """
            this method use shape_fit method's actor(user)-specific blendshapes result.
            neutral pose : user-specific neutral pose
            blendshapes : user specific blendshapes
            ===========================================================================
            return
                weights that are extracted from user-specific blendshapes.
        """

        global lmk_idx
        # define user-specific identity weight and expression.
        lmk_idx = np.array(lmk_idx)


        v_size, dim  = self.neutral_bar.shape
        
        exprs_num, _, _ = self.exprs_bar.shape
        if shape_lmk_idx is None : 
            output_expr_ = self.exprs_bar
            output_neutral_ = self.neutral_bar
        else:
            output_expr_ = self.exprs[:, shape_lmk_idx, :]
            output_neutral_ = self.neutral[shape_lmk_idx, :]

        def coordinate_descent_LBFGS(cost_function, init_x, y, iter_nums, eps = 10e-7, clip_func = None, camera_refinement_func = None ,  contour_mapper_func = None, **kwargs):
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
                if options.get("verbose", False):
                    Rt= get_Rt(*w[-6:, :].ravel())
                    w = w[:-6, :]

                    img = kwargs.get("img", None)
                    mesh = kwargs.get("mesh", None)
                    if img is not None :
                        Q = mesh.get("Q", None )
                        neutral = mesh.get("neutral", None)
                        exprs = mesh.get("exprs", None)
                        f = mesh.get("f", None)
                        m_color = mesh.get("color", (0.5, 0, 0))
                        pts_3d = get_combine_bar_model(neutral, None, exprs,None, w )
                        pts_2d = add_Rt_to_pts(Q, Rt, pts_3d)

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
                                prev_pts3d = get_combine_bar_model(neutral, None, exprs,None, w )
                                prev_pts_2d = add_Rt_to_pts(Q, Rt, prev_pts3d)
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
                    
                return wrapper
            
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
                    start_iteriter = 5
                    iteriter  = start_iteriter
                    if kwargs.get("verbose", False):
                        img = np.copy(kwargs.get("img", None))
                        img = vis.resize_img(img, 1000)
                        vis.set_delay(1000)
                        vis.show("Rt fit",img)
                    while iteriter:
                        if iter_i == 0  and iteriter == start_iteriter:
                            
                            rx, ry, rz, tx, ty, tz = camera_refinement_func(x[:slide], y)
                            x[slide:, :] = np.array([rx, ry, rz, tx, ty, tz]).reshape(-1,1)

                        else:
                            rx, ry, rz, tx, ty, tz = camera_refinement_func(x[:slide], y)
                            x[slide:, :] = np.array([rx, ry, rz, tx, ty, tz]).reshape(-1,1)

                        if contour_mapper_func is not None :
                            new_lmk_indx = contour_mapper_func(x) # do not modify its values.
                            new_lmk_idx_history.append(new_lmk_indx)
                        if kwargs.get("verbose", False):
                            img = kwargs.get("img", None)
                            mesh = kwargs.get("mesh", None)
                            Q = mesh.get("Q", None )
                            neutral = mesh.get("neutral", None)
                            exprs = mesh.get("exprs", None)
                            f = mesh.get("f", None)
                            m_color = mesh.get("color", (0.5, 0, 0))
                            aaa= x[slide:, :].ravel()
                            ig = vis.draw_mesh_to_img(img, Q, get_Rt(*x[slide:, :].ravel()), get_combine_bar_model(neutral, None, exprs, None, x[:slide, :]), f, m_color, 1000, "iter : {}".format(start_iteriter +1 -iteriter))
                            vis.set_delay(100)
                            vis.show("Rt fit",ig)
                        iteriter -= 1
                for i in range(len(x[:slide])):
                    xi = x[i, 0]
                    
                    sel_ind_cost_f = cost_function_builder(i, x)
                    
                    f_val = sel_ind_cost_f(xi)
                    sel_idx_grad_func = cost_grad_wrapper(i, x)
                    
                    coord_grad = sel_idx_grad_func(xi).T

                    res = opt.minimize(fun=sel_ind_cost_f, x0 = np.array([[xi]]), jac = sel_idx_grad_func, method="L-BFGS-B", bounds = w_bounds, options=options)
                    x[i, :] = res.x





                if kwargs.get("verbose", False ) :
                    if len(new_lmk_idx_history) == 1:
                        verbose_function(x, **{"verbose" : kwargs.get('verbose', False),"prev_lmk_idx" :new_lmk_idx_history[-1]})
                    else:
                        verbose_function(x, **{"verbose" : kwargs.get('verbose', False),"prev_lmk_idx" :new_lmk_idx_history[-2]})
                        



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



        def get_combine_bar_model(neutral_bar, ids_bar=None, expr_bar=None, w_i=None, w_e=None ):
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
    

        def clip_function(x):
            x[:-6, :] = np.clip(x[:-6, :], a_min=0.0, a_max = 1.0)
            x[-6:-3, :] %= 2*np.pi
            return x 
        
        def add_Rt_to_pts( Q, Rt, x):
            R = Rt[:3,:3]
            t = Rt[:, -1, None]
            xt = x.T
            Rx = R @ xt 
            Rxt = Rx+t
            pj_Rxt = Q @ Rxt
            res = pj_Rxt/pj_Rxt[-1, :]
            return res[:2, :].T
            
        def expression_cost_funcion(Q, Rt, neutral_bar, exprs_bar, expr_weight, y):
            #alpha star is predefined face_mesh.
            # x := Q + Rt + expr weight

            blended_pose = get_combine_bar_model( neutral_bar=neutral_bar, ids_bar =  None, expr_bar = exprs_bar, w_i = None, w_e = expr_weight)
            
            gen = add_Rt_to_pts(Q, Rt, blended_pose)
            gen = geo.convert_to_cv_image_coord(gen, h)
            z = gen - y
            new_z = z.reshape(-1, 1)
            new_z = new_z.T @ new_z 

            return new_z
        def decompose_Rt( Rt):
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
        def find_camera_parameter_by_cv2(pts3d : np.ndarray,  pts2d : np.ndarray, guessed_Q = None , init_Rt = None):
            """
                pts2d
                pts3d 
                size tuple

            """
            
            if init_Rt is not None : 
                rvec = np.array(self.decompose_Rt(init_Rt))
                tvec = init_Rt[:, -1, np.newaxis]

                succ, rvec, tvec = cv2.solvePnP(pts3d, pts2d, cameraMatrix= guessed_Q, distCoeffs=np.zeros((4), dtype=np.float32), rvec=rvec, tvec=tvec, useExtrinsicGuess=True)
            else :
                succ, rvec, tvec = cv2.solvePnP(pts3d, pts2d, cameraMatrix=guessed_Q, distCoeffs=np.zeros((4), dtype=np.float32))
            res = np.zeros((3,4))
            res[:, :-1] = cv2.Rodrigues(rvec)[0]
            res[:, -1] = tvec.ravel()
            return res
        def camera_posit_func_builder(Q, neutral_bar ,exprs_bar):
            def camera_posit_func(expr_weight, pts2d):
                pts3d = get_combine_bar_model( neutral_bar=neutral_bar, ids_bar =  None, expr_bar = exprs_bar, w_i = None, w_e = expr_weight)
                # new_Q, Rt = self.find_camera_matrix(pts3d, pts2d)
                Rt = find_camera_parameter_by_cv2(pts3d, pts2d, guessed_Q= Q)
                rx, ry, rz = decompose_Rt(Rt)
                tvec = Rt[:, -1].reshape(-1,1)
                return rx,ry,rz, tvec[0, 0 ], tvec[1, 0], tvec[2, 0]
            return camera_posit_func
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
        def cost_function_builder(Q, neutral_bar, expr_bar):
            def wrapper(x, y):
                w_e = x[:-6, 0]
                Rt= get_Rt(*x[-6:,0].ravel())
                return expression_cost_funcion(Q, Rt, neutral_bar, expr_bar, w_e, y)
            return wrapper

        init_weight = np.ones((exprs_num+6, 1), dtype=np.float32) * 0.4

        cost_f = cost_function_builder(Q, self.neutral_bar, self.exprs_bar)
        camera_func = camera_posit_func_builder(Q, self.neutral_bar, self.exprs_bar)
        
        res, _, _ = coordinate_descent_LBFGS(cost_f,
                                        init_weight, lmk2d, iter_nums = 100 ,clip_func = clip_function, 
                                        camera_refinement_func= camera_func,
                                        # **{"verbose":True,
                                        #                     "img" : kwargs.get("img"), \
                                        #                     "lmk_idx": kwargs.get("lmk_idx"), \
                                        #                     "mesh": {"Q" : Q, "color" :(0.8,0,0), "neutral": self.neutral,"exprs": self.exprs, "f" : self.face}, \
                                        #                     "x_info": {"pts":(255,0,0), "line":(255,255,0)},\
                                        #                     "y_info": {"pts":(0,0,255), "line":(0,255,255)},\
                                        #                     "width":1000}\
                                        )
        

        expr_weight = res[:-6, :]
        
        cam_weight = res[-6:, :]
        new_Rt = get_Rt(*cam_weight.ravel())
        res = get_combine_bar_model(output_neutral_, None, output_expr_, None ,expr_weight)

        return res, new_Rt



    
    
    
    def load_model(self, path):
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



        import dlib
        pred_path = "./shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(pred_path)

        tmp_root_path = "cd_test"
        self.neutral, self.face = igl.read_triangle_mesh(os.path.join(tmp_root_path, "gen_identity.obj"))
        self.neutral_bar = self.neutral[lmk_idx, :]
        import glob
        paths = glob.glob(os.path.join(tmp_root_path, "gen_expression_*.obj"))
        paths.sort(key = natural_keys)
        exprs = []
        for p in paths:
            v, _ = igl.read_triangle_mesh(p)
            exprs.append(v)
        self.exprs = np.array(exprs)
        self.exprs -= np.expand_dims(self.neutral, axis=0)
        self.exprs_bar = self.exprs[:,lmk_idx, :]


        with open(os.path.join(path, "meta.txt"), 'r') as fp :
            meta = yaml.load(fp, yaml.FullLoader)
            self.T = meta['T']
            self.K = meta['K']
            self.F = meta['F']
            self.Q = np.load(os.path.join(self.save_path, meta['Q']))
            # self.mean_shape = np.load(os.path.join(self.save_path, meta['mean_shape']))
            self.S_list = np.load(os.path.join(self.save_path, meta['S']))
            self.P = meta['P']
            self.beta = meta['beta']
            self.lmk_size = meta['lmk_size']
        
        self.weak_regressors = [FirstLevelFern(self.Q, self.K, self.F, self.beta, self.P) for _ in range(self.T)]
        for weak in tqdm.tqdm(self.weak_regressors):
            weak.load(path)
        

    @staticmethod
    def split_data(train_data_collection):
        def _wrapper(i):
            image = train_data_collection[i]['img']
            M = train_data_collection[i]['Rt_inv_index']
            S_init_pose = train_data_collection[i]['S_init']
            S = train_data_collection[i]['S_index']
            return image, M, S_init_pose, S
        return _wrapper
    
    @staticmethod
    def normalize_matrix(mean_shape, shape):
        """
            shape to mean
            return scale, rot comibned matrix
        """
        RR, _= scipy.linalg.orthogonal_procrustes(shape - np.mean(shape, -1, keepdims=True), mean_shape - np.mean(mean_shape, -1, keepdims=True))

        return RR
        
        # # initial shapes.
        # b = mean_shape.reshape(-1,1)
        # A = np.zeros((3*len(mean_shape), 9))
        # for j, raw in enumerate(shape):
        #     A[j*3, :3] = raw 
        #     A[j*3+1, 3:6] = raw 
        #     A[j*3+2, 6:] = raw
        # return np.linalg.lstsq(A, b)[0].reshape(3,3)
            


    def train(self, train_data_collection : list, neutral_v : np.ndarray, Ss : list, Rt_invs : list):
        """
            train_data_collection : list of  triandata 
            [Image, Rt, 3d mesh pts, initial 3d mesh pts]
            [I_i, M_i, S_i, S^init_i]
        """
        
        self.weak_regressors = [FirstLevelFern(self.Q, self.K, self.F, self.beta, self.P, name=iid) for iid in range(self.T)]
        from itertools import repeat


        N = len(train_data_collection) # training dataset n*m*G*H = 60*9*5*4 = 10,800
        split_data = TwoLevelBoostRegressor.split_data(train_data_collection)
            
        image, M, S_init_pose, S_index = split_data(0)
        if len(image.shape) == 2 : 
            h, w = image.shape
        else : 
            h, w , _ = image.shape

        self.lmk_size, dim = Ss[S_index].shape
        
        self.lmk_shapes = (self.lmk_size, dim)
        S_size = Ss[S_index].size

        #preprocessing
        image_list = []
        M_data = np.array(Rt_invs)
        M_index_list = []
        M_sorted_list = []
        self.S_list = np.array(Ss)
        S_init_list = []
        self.S_index_list = []
        for i in tqdm.trange(N,desc="data split..."):
            image, M_index, S_init_pose, S_index = split_data(i)
            image_list.append(image)
            M_index_list.append(M_index)
            self.S_index_list.append(M_index)
            M_sorted_list.append(M_data[M_index])
            S_init_list.append(S_init_pose)
        
        S_init_list = list(map(lambda idx : self.S_list[idx], S_init_list))

        # S_list = np.array(self.S_list)
        
        self.mean_shape = np.zeros_like(self.S_list[0])
        # S_list N x K x 3
        # K x 3
        # means = np.mean(self.S_list, axis=1, keepdims=True)
        # centered_shape = self.S_list - np.mean(self.S_list, axis=1, keepdims=True)
        centered_shape = self.S_list
        # centered_shape = self.S_list 
        scale = np.sqrt(np.mean(np.sum(np.power(centered_shape, 2.0), axis=-1), axis=-1))

        # self.mean_shape = np.mean(centered_shape, axis=0)
        self.mean_shape = np.mean(centered_shape, axis=0)


        # self.mean_shape = np.mean(centered_shape/scale.reshape(-1,1,1), axis=0)
        # self.mean_shape = np.mean(centered_shape/scale.reshape(-1,1,1), axis=0)

        # self.mean_shape = self.nuetral_v


        #mean_shape vis
        # mea_v1 = proj(Q, np.eye(3,4), self.mean_shape1)
        mea_v2 = proj(Q, np.eye(3,4), self.mean_shape)
        
        #TODO for coord test.
        # mea_v1 = geo.convert_to_cv_image_coord(mea_v1, h)
        mea_v2 = geo.convert_to_cv_image_coord(mea_v2, h)

        # iia1 = vis.draw_circle(mea_v1, image, colors=(0,0,1), radius=1)
        iia2 = vis.draw_circle(mea_v2, image, colors=(1,0,0), radius=1)
        iia = iia2
        # iia = vis.concatenate_img(1,2,iia1,iia2)
        
        vis.set_delay(100)
        vis.show("test", iia)

        # np.save(os.path.join(self.save_path, "S"), self.S_list)
        global multithread_flag 
        pre_init_weak_f = functools.partial(FirstLevelFern.pre_initialize, data_size = N, lmk_size = self.lmk_size)
        # precompute_calc_offset_d_f = functools.partial(FirstLevelFern.calc_offset_d, Q=self.Q, samples=self.mean_shape, P=self.P, w =w, h= h, kappa=1.0 )
        try : 

            multithread_flag_ = multithread_flag
        except:
            multithread_flag_ = False
        
        if multithread_flag_ : 
            pass
            # mt_pool.starmap(FirstLevelFern.calc_offset_d, zip(self.weak_regressors, repeat(self.Q), repeat(self.mean_shape), repeat(self.P), repeat(w), repeat(h)))
            # mt_pool.map(FirstLevelFern.calc_offset_d, zip(self.weak_regressors, repeat(self.Q), repeat(self.mean_shape), repeat(self.P), repeat(w), repeat(h)))

        for weak in self.weak_regressors:
            pre_init_weak_f(weak)
            # precompute_calc_offset_d_f(weak)
            
        # mt_pool.map(pre_init_weak_f, self.weak_regressors)



        # current_shapes = np.zeros((len(S_init_list),self.lmk_size, 3), dtype=np.float32)
        current_shapes = np.array(S_init_list, copy=True)
        # for Si, (S_init) in tqdm.tqdm(enumerate(S_init_list), desc="init current shapes."):
            # current_shapes[Si, ...] = S_init
        # current_shapes[...] = S_init_list 
        

        regression_targets = np.zeros_like(current_shapes)
        # image_list_test = [image_list[-1]]

        shape_indices = list(range(len(current_shapes)))
        regression_targets = np.zeros_like(current_shapes)
        GT_S_List = np.zeros_like(current_shapes)
        kappa = 1.0
        for weak_regressor in tqdm.tqdm(self.weak_regressors, desc="weak regressor train mode."):
            # for i, shape in enumerate(current_shapes):
            #     image, M, S, cur_shape = image_list[i], M_data[ self.S_index_list[i] ], \
            #                             Ss[self.S_index_list[i]], current_shapes[i]
            #     regression_targets[i, ...] =  (S - cur_shape)
            #     GT_S_List[i,...] = S
            regression_targets[ shape_indices , ... ] = Ss[self.S_index_list] - current_shapes

            GT_S_List[shape_indices, ...] = Ss[self.S_index_list]
            
            # pred = weak_regressor.train(regression_targets, image_list, current_shapes, Ss, self.S_index_list, M_sorted_list, self.mean_shape,Gt=GT_S_List)
            pred = weak_regressor.train(regression_targets, image_list, current_shapes, Ss, self.S_index_list, M_sorted_list, self.mean_shape,Gt=GT_S_List, kappa=kappa)
            kappa -= 0.1
            current_shapes += pred


        
        # self.save_img(image_list, current_shapes, Ss[self.S_index_list], M_sorted_list)
        self.save_model()
    def save_img(self, images, cur_shape, Ss, M_list):
        path = "./result_img"
        if not os.path.exists(path):
            os.makedirs(path)
        for i, (img, cur, s, M) in enumerate(zip(images, cur_shape, Ss, M_list)):
            cur = proj(self.Q, M, cur)
            s = proj(self.Q, M, s)
            img = cv2.cvtColor(np.copy(img), cv2.COLOR_GRAY2RGB)
            im = vis.draw_circle(cur, img, (0,0,255), radius=1)
            im = vis.draw_circle(s, im, (255,0,0),radius = 1)
            vis.save(osp.join(path, str(i)+".png"), im)


    def predict(self, o, init_num = 15, prev_data = None, render= False, lmk_idx=None, Q = None ):
        """
         o : list of images or image 
        """
      

        if isinstance(o, list):
            res = []
            for item in o :
                res_item = self._predict(item)
                # cv2.imshow("test", res_item)
                # cv2.waitKey(100)
                # res.append(res_item, init_num)
            return res 
        elif isinstance(o, np.ndarray):
            # while True:
            #     data, new_Rt =self._predict(o, init_num, prev_frame_S=prev_data, clmk_idx = lmk_idx)
            #     # Rtinv = TwoLevelBoostRegressor.inverse_Rt(new_Rt)
            #     Rtinv = new_Rt
            #     if render :
            #         if len(o.shape) == 3:
            #             h,w,c =(o.shape)
            #         else :
            #             h,w = o.shape
            #         res = proj(self.Q, Rtinv, data)

            #         im = vis.draw_circle(res, o, colors=(0,0,255), radius=1)
            #         im = vis.resize_img(im, 1000)
            #         vis.set_delay(10)
            #         vis.show("result", im)
            #         prev_data = (Rtinv[:, :3] @data.T + Rtinv[:, -1, None]).T
            data, new_Rt =self._predict(o, init_num, prev_frame_S=prev_data, clmk_idx = lmk_idx,  Q = Q if Q is not None else None)
            # Rtinv = TwoLevelBoostRegressor.inverse_Rt(new_Rt)
            Rtinv = new_Rt
            if render :
                h,w,*_ =(o.shape)
                # res = proj(self.Q, Rtinv, data)

                # im = vis.draw_circle(res, o, colors=(0,0,255), radius=1)
                # im = vis.resize_img(im, 1000)
                # vis.set_delay(10)
                # vis.show("result", im)
            prev_data = add_to_pts(Rtinv, data)
                # prev_data = (Rtinv[:, :3] @data.T + Rtinv[:, -1, None]).T
            return prev_data
            
        else : 
            raise TypeError(repr(type(o))+"it is not list of images or image(ndarray)")
    
    @staticmethod
    def similarity_transform(src, dest):
        """
            Ruturn Rt that src to dest
        """


        src_mean = np.mean(src, axis=0)
        dest_mean = np.mean(dest, axis=0)
        centered_src = src - src_mean
        centered_dest = dest- dest_mean
        
        RR, s = scipy.linalg.orthogonal_procrustes(centered_src, centered_dest)
        trans_center_to_dest = dest_mean
        Rt = np.eye(4,4, dtype=np.float32)
        new_Rt = np.zeros((3,4), dtype=np.float32)
        translate_center = np.eye(4,4, dtype=np.float32)
        translate_center[-1,-1] = 1
        translate_center[:-1, -1, None] -= src_mean.reshape(-1, 1)
        Rt[:3,:3] = RR
        Rt[:-1, -1, None ] = trans_center_to_dest.reshape(-1,1)
        tmp = Rt @ translate_center
        new_Rt[:, :] = tmp[:3, :]
        # A = np.zeros((3*len(src), 9), dtype=np.float32)

        # for i in range(len(centered_src)):
        #     A[i*3   , :3]  = centered_src[i]
        #     A[i*3+1 , 3:6]  = centered_src[i]
        #     A[i*3+2 , 6:]  = centered_src[i]
        
        # flat_S = centered_dest.reshape(-1, 1)
        
        # res = np.linalg.lstsq(A, flat_S)

        # R = res[0].reshape(3,3)
        # t = dest_mean - np.mean((R @ src.T).T, axis=0)
        # t =t.reshape(-1,1)
        # Rt= np.concatenate([R,t],axis=-1)
        return new_Rt

    @staticmethod
    def inverse_Rt(Rt):
        RR = Rt[:3,:3]
        Rinv = RR.T
        invt = - Rinv @ Rt[:, -1, None]
        Rtinv = np.zeros_like(Rt)
        Rtinv[:, :3] = Rinv ; Rtinv[:, -1] = invt.reshape(-1)
        return Rtinv
        

    def find_most_similar_to_target_pose(self, pose_collection, target_pose, candidate_num = 20):
        #https://stackoverflow.com/questions/6422700/how-to-get-indices-of-a-sorted-array-in-python
        # ss = [i for i, _ in sorted(enumerate(pose_collection), key = lambda S : np.mean(np.sum(((S[1] - target_pose)**2), axis=-1)))]

        def distance_f1(S1, S2): # cloestst distance with centered mesh 
            # S, R, SRt, Rt, *_ = similarity_transform2(S1, S2)
            # res = add_to_pts(SRt, S1)
            # return np.sum(( S1 - S2 )**2)
            return np.sum(( (S1 - np.mean(S1, axis=0)) - (S2 - np.mean(S2,axis=0) )**2))

        def distance_f2(S1, S2): #distance between similar shapes.
            return np.sum((np.mean(S1, axis=0)  - np.mean(S2, axis=0))**2)
        # sorted_ss = [i for i, _ in sorted(enumerate(pose_collection), key = lambda S : (np.sum((( (S[1] - np.mean(S[1], axis=0)) - (target_pose - np.mean(target_pose, axis=0)))**2))))]
        sorted_ss = [i for i, _ in sorted(enumerate(pose_collection), key = lambda S :  distance_f1(S[1] , target_pose))]
        # sorted_sps2 = [i for i in sorted(sorted_ss, key = lambda S :  distance_f2(pose_collection[S] , target_pose))]
        # ss_distance = [i for i, _ in sorted(enumerate(pose_collection), key = lambda S : (np.sum((( np.mean(S[1], axis=0) - np.mean(target_pose, axis=0) )**2))))]

        return sorted_ss[:candidate_num]

    def _predict(self, img, init_num, prev_frame_S = None, clmk_idx = None, Q = None):
        vis.set_delay(1000)
        if Q is not None :
            self.Q = Q
        ####    for testing 
        self.mean_shape = np.zeros_like(self.S_list[0])
        # S_list N x K x 3
        new_Rt = np.eye(3,4)
        # K x 3
        self.S_list = np.array(self.S_list)
        means = np.mean(self.S_list, axis=1, keepdims=True)
        # centered_shape = self.S_list - np.mean(self.S_list, axis=1, keepdims=True)
        centered_shape = self.S_list 
        # centered_shape = self.S_list
        scale = np.sqrt(np.mean(np.sum(np.power(centered_shape, 2.0), axis=-1), axis=-1))

        # self.mean_shape = np.mean(centered_shape/scale.reshape(-1,1,1), axis=0)
        self.mean_shape = np.mean(centered_shape, axis=0)
        # self.mean_shape = self.nuetral_v
        def get_image_shape( img):
            (h, w, *_) = img.shape
            return h, w
        # def split_data(train_data_collection):
        # def _wrapper(i):
        #     image = train_data_collection[i]['img']
        #     M = train_data_collection[i]['Rt_inv_index']
        #     S_init_pose = train_data_collection[i]['S_init']
        #     S = train_data_collection[i]['S_index']
        #     return image, M, S_init_pose, S
        # return _wrapper
        #preprocessing

        # split_data = TwoLevelBoostRegressor.split_data(self.test_data) #for testing

        # S_init_list = []
        # for i in tqdm.trange(len(self.test_data),desc="data split..."):
        #     _,_, S_init_pose, _ = split_data(i)
        #     S_init_list.append(S_init_pose)
        # S_init_list = list(map(lambda idx : self.S_list[idx], S_init_list))
        #####
        result = np.zeros((self.lmk_size, 3), dtype=np.float32)
        prev_ind = -1
        h, _ = get_image_shape(img)
        if prev_frame_S is None : 
            

            prev_frame_S2d = self.detect_by_dlib(img)
            if clmk_idx is None :
                clmk_idx = lmk_idx
            im = vis.draw_circle(prev_frame_S2d, img, colors=(0,0,255))
            im = vis.resize_img(im, 1000)
            vis.show(title="test",img=im)


            prev_frame_S2d = geo.convert_to_camera_uv_coord(prev_frame_S2d,h)

            prev_frame_S, new_Rt = self.refine_3d_lmk(prev_frame_S2d, img=img, shape_lmk_idx=clmk_idx)
            prev_frame_S = (new_Rt @ np.concatenate([prev_frame_S, np.ones((len(prev_frame_S),1),dtype=np.float32)], axis=-1).T).T

        # find most nearest shape to prev_frame_S in original shape space 
        loss = np.inf 
        for i, S in enumerate(self.S_original):
            s_loss = (np.sum((((S - np.mean(S, axis=0)) - (prev_frame_S - np.mean(prev_frame_S, axis=0)))**2)))
            # s_loss = np.sum(((S - prev_frame_S)**2))
            if s_loss < loss: 
                loss = s_loss
                ind = i
        # S_r = self.S_list[ind]
        S_r = self.S_original[ind]
        
        
        S_Rp = proj(self.Q,  np.eye(3,4), S_r)
        S_Rp = geo.convert_to_cv_image_coord(S_Rp, h)
        im = vis.draw_circle(S_Rp, img, colors=(0,0,255), radius=2)
        res = proj(self.Q,  np.eye(3,4), prev_frame_S)
        res = geo.convert_to_cv_image_coord(res, h)
        im = vis.draw_circle(res, im, colors=(0,255,0), radius=2)
        im = vis.resize_img(im, 1000)
        vis.show(title="tes2t",img=im)
        trained_img_size = 1000 # for testing
        if len(img.shape) == 3 : 
            intensity_img = cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY)
        else:
            intensity_img = img
        
        #TODO thi is rotation problem ' it is not ortho
        # new_Rt = self.similarity_transform(prev_frame_S, S_r) 
        Scale, R, trans_vec, scaled_combined_Rt,combined_Rt, inv_RR  = similarity_transform2(prev_frame_S, S_r)
        
        # prev_frame_S = np.concatenate([prev_frame_S, np.ones((len(prev_frame_S), 1))], axis=-1)
        # prev_frame_S = (new_Rt@prev_frame_S.T).T   
        prev_frame_S_no_rt = prev_frame_S
        prev_frame_S = add_to_pts(combined_Rt[:3,:], prev_frame_S)     
        # prev_frame_S = add_to_pts(Scale*R, prev_frame_S)     
        # im = vis.draw_circle(res, img, colors=(0,255,0))
        res = proj(self.Q,  np.eye(3,4), prev_frame_S)
        S_sim_proj = proj(self.Q,  np.eye(3,4), S_r)
        # im = vis.draw_circle(res, img, colors=(0,255,0))
        # im = vis.draw_circle(S_Rp, im, colors=(0,0,255))

        # im = vis.draw_circle(res, img, colors=(255,0,0), radius=2)
        # im = vis.draw_circle(S_sim_proj, im, colors=(0,0,255), radius=2)
        # im = vis.resize_img(im, 1000)
        # vis.show(title="S_RandprevS",img=im)
        # indices = self.find_most_similar_to_target_pose(pose_collection= self.S_list, \
        indices = self.find_most_similar_to_target_pose(pose_collection= self.S_list, \
                                                            target_pose = prev_frame_S, \
                                                            candidate_num=init_num )
        
        # im = np.copy(img)
        # for S in self.S_list:
        #     rr = S
        #     im = vis.draw_circle(proj(self.Q, np.eye(3,4), rr), im, (0,255,255))
        #     vis.show("full_shapes", vis.resize_img(im,1000))

        # for ii in indices:
        #     im = np.copy(img)
        #     im2 = np.copy(img)
        #     # rr = self.S_list[ii]
        #     rr = self.S_list[ii]
        #     im = vis.draw_circle(res, im, (255,0,0), radius=1)
        #     im = vis.draw_circle(proj(self.Q, np.eye(3,4), rr), im, (0,0,255), radius=1)
        #     im = vis.resize_img(im,600)
        #     im2 = vis.draw_circle(res, im2, (255,0,0), radius=1)
        #     im2 = vis.draw_circle(proj(self.Q, InvRt, rr), im2, (0,255,255), radius=1)
        #     imim = vis.concatenate_img(1,2, im,im2)
        #     vis.show("candidate_yello", vis.resize_img(imim,1000))


        result = np.zeros_like(prev_frame_S)
        InvRt =TwoLevelBoostRegressor.inverse_Rt(combined_Rt[:3, :])
        #inverse checking
        # aab = np.arange(1*3).reshape(-1,1)
        # t_aab = add_to_pts(combined_Rt[:3,:], aab)
        # print(add_to_pts(inv_RR[:3,:], t_aab))
        # print(add_to_pts(InvRt[:3,:], t_aab))
        inv_RR = InvRt
        for l in indices:
            cur_pose = np.copy(self.S_list[l])
            # cur_pose = np.copy(S_init_list[l])
            # im = vis.draw_circle(proj(self.Q, InvRt, cur_pose), img, (0,0,255), radius=1)
            # cur_pose = np.concatenate([cur_pose, np.zeros((len(cur_pose), 1))], axis=-1)
            # cur_pose = (new_Rt@cur_pose.T).T
            # im = vis.draw_circle(proj(self.Q, combined_Rt[:3,:], cur_pose), im, (0,255,0))

            # for weak_regressor in tqdm.tqdm(self.weak_regressors, "weak"):
            for weak_regressor in self.weak_regressors:
                # cur_pose = weak_regressor.predict(intensity_img, cur_pose, S_prime=prev_frame_S,mean_shape=self.mean_shape, predRt = InvRt, Q = self.Q)
                cur_pose += weak_regressor.predict(intensity_img, cur_pose, S_prime=prev_frame_S,mean_shape=self.mean_shape, predRt = inv_RR[:3,:], Q = self.Q)
                # pred = weak_regressor.predict(intensity_img, cur_pose, S_prime=prev_frame_S,mean_shape=self.mean_shape, predRt = combined_Rt[:3,:], Q = self.Q)
                # cur_pose += pred 

            # im = vis.draw_circle(proj(self.Q, InvRt, cur_pose), img, (255,255,0), radius=1)
            # vis.show("test", vis.resize_img(im,1000))
            result += cur_pose
        result = result / init_num

        testing = proj(self.Q, InvRt, result)
        pp = proj(self.Q, InvRt, result)
        pp = geo.convert_to_cv_image_coord(pp, h)
        im = vis.draw_circle(pp, img, (255,255,0), radius=1)
        # im = vis.draw_circle(proj(self.Q, np.eye(3,4), result), im, (0,255,255), radius=1)
        vis.show("result", vis.resize_img(im,1000))

        # return result, scaled_combined_Rt[:3, :]
        return result, inv_RR[:3,:]
                
            

def load_data_train(path, start = None, end = None, lmk_indices = None, desired_img_width = None):
    meta_path = os.path.join(path, "meta.txt")
    with open(meta_path, 'r') as fp :
        meta = yaml.load(fp, yaml.FullLoader)
        image_ext = meta['image_extension']
        image_root = meta["image_root_location"]
        S_Rtinv_index_list_path = meta["S_Rtinv_index_list"]
        Q = np.load(os.path.join(path,meta['Q_location']))
        image_location = meta['image_name_location']
        Rt_inv_location = meta['Rt_inv_location']
        S_init_location = meta['S_init_location']
        S_location = meta['S_location']
        S_original_location = meta['S_original_location']
        root_path = meta['data_root']

    data_root = os.path.join(path, root_path)
    S_Rtinv_index_list = np.load(os.path.join(data_root,S_Rtinv_index_list_path))
    img_names = np.load( os.path.join(data_root, image_location))
    Ss = np.load(os.path.join(data_root, S_location))
    S_original = np.load(os.path.join(data_root, S_original_location))
    S_inits = np.load(os.path.join(data_root, S_init_location))
    Rt_invs = np.load(os.path.join(data_root, Rt_inv_location))
    res = [] 

    image_dict = {}
    img_names2 = list(set(img_names))
    resize_ratio = 1.0
    for name in img_names2:
        img = cv2.imread(os.path.join(image_root, name+image_ext ))
        if desired_img_width != None :
            h, w, _ = img.shape
            resize_ratio = desired_img_width/w
            img = cv2.resize(img, (int(resize_ratio*w),int(resize_ratio*h)))
        image_dict[name] = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img)

    if lmk_indices == None :
        lmk_indices = list(range(Ss.shape[1]))

    Ss = np.array(Ss)[: , lmk_indices, :]

    # for img_name, s, s_init, Rt_inv in tqdm.tqdm(zip(img_names, Ss, S_inits, Rt_invs), "load data"):
    for img_name,  s_init, S_Rt_index in tqdm.tqdm(zip(img_names, S_inits, S_Rtinv_index_list), "load data"):
        # res.append({'img' : image_dict[img_name][0], 'color_img' : image_dict[img_name][1], "S_index" : S_Rt_index, "S_init" : s_init[lmk_indices, :], "Rt_inv_index" : S_Rt_index})
        res.append({'img' : image_dict[img_name][0], 'color_img' : image_dict[img_name][1], "S_index" : S_Rt_index, "S_init" : s_init, "Rt_inv_index" : S_Rt_index})
    return Q, res[start:end], Ss, S_original, Rt_invs, resize_ratio
    



def get_kinect_Q():
    sensor_size = 3.1 * 0.001 # um
    fov_h = np.deg2rad(70/2.0)
    fov_v = np.deg2rad(60/2.0)
    
    # mm
    focal_length_x = 3.2813/sensor_size
    focal_length_y = 3.5157/sensor_size
    #pixel
    principal_x = 965.112
    principal_y = 583.267
    width_ratio = 640/1920

    #this is kinect Q
    Q = np.array([[focal_length_x,0,principal_x],[0,focal_length_y,principal_y],[0,0,1]], dtype=np.float64)
    return Q
if __name__ == "__main__":
    # Q, data, Ss, Rt_invs, resize_ratio = load_data_train("./cd_test/", start=0, end=5, desired_img_width=800, lmk_indices=lmk_without_contour_idx)
    # Q, data, Ss, Rt_invs, resize_ratio = load_data_train("./cd_test/", desired_img_width=800, lmk_indices=lmk_without_contour_idx)
    # new_indices = list(range(lmk_without_contour_idx[0], len(regression_inner_contour_indices)+len(lmk_idx)))
    # Q, data, Ss, S_original, Rt_invs, resize_ratio = load_data_train("./cd_test/", desired_img_width=640)
    # Q, data, Ss, S_original, Rt_invs, resize_ratio = load_data_train("./kinect_dataset/")
    Q, data, Ss, S_original, Rt_invs, resize_ratio = load_data_train("./kinect_test_dataset/")
    # Q, data, Ss, S_original, Rt_invs, resize_ratio = load_data_train("./cd_test/")
    # Q, data, Ss, Rt_invs = load_data_train("./cd_test/", start=-4, end=-1)
    # Q, data, Ss, Rt_invs, resize_ratio = load_data_train("./cd_test/", lmk_indices=lmk_without_contour_idx)
    

    Q = np.array([[2.29360512e+03, 0.00000000e+00, 2.61489110e+03],
        [0.00000000e+00, 2.29936264e+03, 1.94713585e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
    Q[:-1, :] *= resize_ratio

    sensor_size = 3.1 * 0.001 # um
    fov_h = np.deg2rad(70/2.0)
    fov_v = np.deg2rad(60/2.0)
    
    # mm
    focal_length_x = 3.2813/sensor_size
    focal_length_y = 3.5157/sensor_size
    #pixel
    principal_x = 965.112
    principal_y = 583.267
    width_ratio = 640/1920

    #this is kinect Q
    Q = np.array([[focal_length_x,0,principal_x],[0,focal_length_y,principal_y],[0,0,1]], dtype=np.float64)
    # kinect_Q[:-1, :] *=width_ratio

    # Q[-1, -1] /= resize_ratio

    # for i in range(len(data)):
    #     img = data[i]["color_img"]
    #     S = data[i]["S_index"]
    #     S_pts = Ss[S]

    #     Rt_inv = data[i]["Rt_inv_index"]
    #     Rt_inv_mat = Rt_invs[Rt_inv]
    #     S_init = data[i]["S_init"]
    #     S_init = Ss[S_init]

    #     S_prj = proj(Q, Rt_inv_mat, S_pts)
    #     S_i_proj = proj(Q, np.eye(3,4, dtype=np.float32), S_init)
    #     new_img = vis.draw_circle(S_prj, img, colors=(0,0,255), radius=1)
    #     im = vis.draw_circle(S_i_proj, img, colors=(255,0,0), radius=2)
    #     im1 = vis.resize_img(im, 600)
    #     im2 = vis.resize_img(new_img, 600)
    #     im = vis.concatenate_img(1, 2, im2, im1)
        

    #     vis.set_delay(100)
    #     vis.show("test", im)

    import igl 
    import os.path as osp
    regression_inner_contour_indices = np.load(osp.join("./predefined_face_weight", "regression_contour.npy"))

    neutral_v, _ = igl.read_triangle_mesh("data/generic_neutral_mesh.obj")
    multithread_flag = True
    # lmk_idx = np.array(lmk_idx, dtype=np.uint)[lmk_without_contour_idx] #TODO
    lmk_new_idx = np.array(lmk_idx, dtype=np.uint)[lmk_without_contour_idx]
    # lmk_new_idx = np.concatenate([lmk_new_idx, regression_inner_contour_indices], axis=-1)
    lmk_new_idx = regression_inner_contour_indices.astype(np.uint)
    # neutral_v = neutral_v[lmk_idx, :]
    neutral_v = neutral_v[lmk_new_idx, :]
    # reg = TwoLevelBoostRegressor(Q = Q,Ss=Ss, S_original = S_original, nuetral_v=neutral_v, data=data)
    # reg = TwoLevelBoostRegressor(Q = Q,beta=250,   Ss=Ss, S_original = S_original, nuetral_v=neutral_v, data=data,P=400)
    reg = TwoLevelBoostRegressor(Q = Q,beta=0, Ss=Ss, S_original = S_original, neutral_v=neutral_v, data=data,P=100)
    # reg = TwoLevelBoostRegressor(Q = Q,T=10, K=10,beta=1000, Ss=Ss, S_original = S_original, nuetral_v=neutral_v, data=data)
    reg.set_save_path("./fern_pretrained_data")
    reg.train(data, neutral_v, Ss, Rt_invs)
    reg.load_model("./fern_pretrained_data")
    prev_data = None 
    prev_data = add_to_pts( Rt_invs[data[0]['Rt_inv_index']], Ss[data[0]['S_index']])
    ida= np.eye(3,4)
    dt = proj(Q, ida, prev_data)
    h, w , _ = data[0]['color_img'].shape
    dt = geo.convert_to_cv_image_coord(dt, h)

    frame = vis.draw_circle(dt, data[0]['color_img'], colors=(0,0,255), radius=1, thickness=3)

    # vis.show("default", frame)

    result = []
    prev_data = None
    # prev_data  = reg.predict(data[0]['color_img'], render = True, lmk_idx = lmk_new_idx, prev_data=prev_data)
    result.append(prev_data)
    for i in range(100):
        
        prev_data2  = reg.predict(data[0]['color_img'], render = True, lmk_idx = lmk_new_idx, prev_data=prev_data)
        result.append(prev_data2)
        prev_data = prev_data2
        dt = proj(Q, ida, prev_data)
        dt = geo.convert_to_cv_image_coord(dt, h)
        
        frame = vis.draw_circle(dt, data[0]['color_img'], colors=(0,0,255), radius=1, thickness=3)
        vis.show("VideoFrame", frame)
    # print(result)
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    prev_data = None 
    ida= np.eye(3,4)
    #https://www.researchgate.net/figure/Kinect-camera-intrinsic-parameters-the-resolution-and-sensor-size-information-from-36_tbl1_321354490
    
    while cv2.waitKey(33) < 0:
        ret, frame = capture.read()
        # frame = vis.resize_img(frame, 640)
        prev_data2 = reg.predict(frame, prev_data=prev_data, lmk_idx = lmk_new_idx, Q = Q)
        
        prev_data = prev_data2
        dt = proj(Q, ida, prev_data2)
        h, w, _ = frame.shape
        dt = geo.convert_to_cv_image_coord(dt, h)
        frame = vis.draw_circle(dt, frame, colors=(0,0,255), radius=1, thickness=3)
        cv2.imshow("VideoFrame", frame)

    capture.release()
    cv2.destroyAllWindows()
    reg.predict(data[-1]['color_img'], render = True, lmk_idx = lmk_new_idx)   


#      We set κ equal to 0.3 times of
# the distance between two pupils on the mean shape.