import numpy as np 

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import yaml 
import cv2 
import tqdm
import time
import visualizer as vis
import multiprocessing
import scipy.spatial as sp 
import igl
import copy
import scipy
import scipy.optimize as opt
import re 



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
    def calc_offset_d(Q, samples, P, w, h):
        """
            pts : N x K
            samples : Vx3
        """
        # R = Ms[:3,:3]
        # t = Ms[:, -1, np.newaxis]


        
        
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
        std = np.std(samples, axis=0)
        factor = 0.7
        while True :
            # disp[idx_list, :] = np.random.normal(np.mean(samples, axis=0) ,(max-min), size = (num_P, 3))
            # disp[idx_list, :] = np.random.uniform(min-1,max+1, size = (num_P, 3))
            disp[idx_list, :] = np.random.uniform(-1, 1, size = (num_P, 3))
            # flag, l = FirstLevelFern.test_outofbound_samples(Q, Ms, disp, w, h)
            flag, l = FirstLevelFern.test_valid_pts(disp)
            if flag: # if good random sample it is .
                break
            else :
                num_P = len(l)
                idx_list = l
                
        nearest_index = list(range(len(samples)))
        randomly_selected_index = list(np.random.randint(low=0, high=len(samples), size=(P - len(samples))))
        nearest_index += randomly_selected_index

        # kdtree =sp.KDTree(samples)
        # disp = np.random.normal(np.mean( samples ,axis=0, keepdims=True), 2,size= (10000, 3))
        # disp[:, -1] = 1
        # _, nearest_index = kdtree.query(disp)
        # disp2d = (Q@(R@disp.T+t)).T
        # disp2d = (Q@(R@disp.T+t)).T
        # for i, nearest_idx in enumerate(nearest_index):
            # disp[i, :] += samples[nearest_idx, :] 
        
        disp2d = (Q@(disp.T)).T
        disp2d = disp2d[:, :-1] / disp[:, -1, np.newaxis]
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
        # ax.scatter(disp[:, 0], disp[:, 1], disp[:, 2], marker="x")
        # ax.scatter(samples[:,0], samples[:,1], samples[:,2], marker="o")
        # plt.show()
        resdisp = np.zeros_like(disp)
        
        # for i, nearest_idx in enumerate(nearest_index):
        #     disp[i, :] -= samples[nearest_idx, :] 
        
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
        pts_2d = proj(Q, M, p)


        # test for draw
        # im = np.copy(Image)
        # import visualizer
        # im = visualizer.draw_circle(pts_2d, im, colors=(255,0,0))
        # im = visualizer.resize_img(im, 1000)
        # visualizer.show("test", im)
        

        if len(Image.shape) >= 3 :
            img_intensity = cv2.cvtColor(Image,  cv2.COLOR_BGR2GRAY)
        else : 
            img_intensity = Image
        # img_intensity = FirstLevelFern.convert_RGB_to_intensity(Image)
        # img_intensity = FirstLevelFern.convert_RGB_to_intensity(Image)
        loc = pts_2d.astype(np.uint)
        loc[loc[:, 0] < 0] = 0
        loc[loc[:, 0] >= w] = w -1 
        loc[loc[:, 1] < 0] = 0
        loc[loc[:, 1] >= h] = h -1 

        intensity_vectors = img_intensity.T[loc[:, 0].ravel(), loc[:, 1].ravel()] # N x 1
        intensity_vectors = intensity_vectors.reshape(-1,1)

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
        cmin = np.min( img, axis = -1 ) # h,w,1
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


    def train(self, regression_targets, image_list, current_shapes, rot_list, rot_inv_list, M_list, mean_shape):
        lmk_size, dim = regression_targets[0].shape
        if len(image_list[0].shape ) == 3 :
            img_h, img_w, _ = image_list[0].shape
        else:
            img_h, img_w = image_list[0].shape
        target_num, lmk_size, dim = regression_targets.shape 

        
        

        N = len(current_shapes) # TODO disable it when real-trainig. 
        # N = 5 # for testing. remove it when you train model. this is toy model
    
        self.V_list = [] # N x P x 1
        self.pixel_loc_list = [] # N x P x 2
        self.nearest_index_list = [] # N x P x 1

        # self.nearset_index, self.disp = FirstLevelFern.calc_offset_d(self.Q, M, neutral_v, self.P, img_w, img_h)
        # self.nearset_index, self.disp = FirstLevelFern.calc_offset_d(self.Q, M, S_init_pose, self.P, img_w, img_h)
        self.nearset_index, self.disp = FirstLevelFern.calc_offset_d(self.Q, mean_shape, self.P, img_w, img_h)
        
        # ig = vis.draw_circle(pixel_loc, image, color=(255,0,0))
        # ig = vis.resize(ig, 1000)
        # vis.show("test", ig)
        for i in range(N): # TODO
            inverted_disp = (np.linalg.inv(rot_inv_list[i]) @ self.disp.T).T
            # inverted_disp = self.disp
            image, M, current_shape = image_list[i], M_list[i],  np.squeeze(current_shapes[i, ...])
            # V_i, pixel_loc, nearest_index, _  = FirstLevelFern.calc_appearance_vector(image,self.Q, M, S_init_pose, self.P, self.disp ,self.nearset_index)
            V_i, pixel_loc, nearest_index, _  = FirstLevelFern.calc_appearance_vector(image,self.Q, M, current_shape, self.P, inverted_disp ,self.nearset_index)
            # self.nearest_index_list.append(nearest_index)
            self.pixel_loc_list.append(pixel_loc)
            self.V_list.append(V_i)

        
        # intensity
        # P x N 
        self.V = np.array(self.V_list, dtype=np.float32).reshape(N, self.P) # N x P x 1
        self.pixel_location = np.array(self.pixel_loc_list)
        # self.nearest_index = np.array(self.nearest_index_list)
        




        # # TODO this is test
        # # pixel diff
        # row = []
        # col = []
        # data = []
        # sparse_row_num = 0 
        # for i in range(self.P):
        #     for j in range(self.P):
        #         if i != j :
        #             row.append(sparse_row_num); col.append(i); data.append(1)
        #             row.append(sparse_row_num); col.append(j); data.append(-1)
        #         sparse_row_num += 1

        # pixel_diff_mat = scipy.sparse.coo_matrix((data, (row, col)), (self.P**2, self.P), dtype=np.float32)
        
        # pixel_diffs = (pixel_diff_mat @ self.V).T


            




        self.P2 = np.zeros((self.P, self.P))
        #calc P2 
        for i in range(self.P):
            for j in range(i, self.P):
                correlation = FirstLevelFern.calc_covariance(self.V[:, i], self.V[:, j])
                self.P2[i, j] = correlation
                self.P2[j, i] = correlation
            
        prediction = np.zeros_like(regression_targets)
        # for ki in tqdm.trange(self.K): #second level regression
        pbar = tqdm.tqdm(self.ferns, desc="train fern(err : inf)", leave=False)
        for fern_regressor in pbar:
            pred = fern_regressor.train(regression_targets, self.P2, self.V, self.pixel_location, self.nearset_index)
            # pred = fern_regressor.train(regression_targets, pixel_diffs, self.V, self.pixel_location, self.nearset_index)
            # for i, p in enumerate(pred):
            #     p[...] = (np.linalg.inv(rot_list[i]) @p.T).T
            prediction += pred 
            regression_targets -= pred
            pbar.set_description("train fern(err : {})".format(np.sum(np.sqrt(np.sum(regression_targets**2, -1))) ))


        return prediction
            

    def predict(self, img, cur_S, S_prime, mean_shape, predRt, Q):
        res = np.zeros_like(cur_S)
        # R = predRt[:,:3]
        # RR = TwoLevelBoostRegressor.normalize_matrix(mean_shape, cur_S)
        Rtinv = TwoLevelBoostRegressor.inverse_Rt(predRt)
        # Rinv = np.linalg.inv(R)
        # invt = - Rinv @ predRt[:, -1, None]
        # Rtinv = np.zeros_like(predRt)
        # Rtinv[:, :3] = Rinv ; Rtinv[:, -1] = invt.reshape(-1)

        intensity_vectors, loc, nearset_index, disp  = FirstLevelFern.calc_appearance_vector(img, Q, Rtinv, cur_S, 400, self.disp, self.nearest_index_list)
        # intensity_vectors, loc, nearset_index, disp  = FirstLevelFern.calc_appearance_vector(img, Q, predRt, cur_S, 400, self.disp, self.nearest_index_list)
        for fern_reg in self.ferns:
            res += fern_reg.pred(img, cur_S, Q, predRt, intensity_vectors)
        return res

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
    def pred(self, img, cur_S, Q,Rt, intensity_vector):
        index = 0
        self.selected_nearest_index = self.selected_nearest_index.astype(np.uint)
        for f in ( range( self.F)):
            nearest_lmk_idx_1 = self.selected_nearest_index[f, 0]
            nearest_lmk_idx_2 = self.selected_nearest_index[f, 1]
            # x = self.selected_pixel_location[f, 0]
            # y = self.selected_pixel_location[f, 1]

            sps = img.shape
            if(len(sps) == 3):
                h, w, _ = img.shape
            else:
                h, w = img.shape


            # shape1 = cur_S[nearest_lmk_idx_1, None]
            # proj_xy = proj(Q, Rt, shape1)
            # proj_xy = (Q @ Rt @ cur_S[nearest_lmk_idx_1, None].T).T
            # proj_xy = proj_xy[:, :-1] / proj_xy[:, -1]
            # proj_xy[proj_xy < 0] = 0
            # proj_xy[proj_xy[:, 0] >= w, 0] = w-1
            # proj_xy[proj_xy[:, 1] >= h, 1] = h-1
            # x,y = proj_xy.astype(np.uint).ravel()
            # intensity_1 = img[y,x].astype(np.float32)
            intensity_1 = intensity_vector[nearest_lmk_idx_1]
            intensity_2 = intensity_vector[nearest_lmk_idx_2]
            
            # shape2 = cur_S[nearest_lmk_idx_2, None]
            # proj_xy = proj(Q, Rt, shape2)
            # proj_xy = (Q@Rt@cur_S[nearest_lmk_idx_2, None].T).T
            # proj_xy = proj_xy[:, :-1] / proj_xy[:, -1]
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
        return self.bin_params[index]



    def train(self, regression_targets_S,P2, V, pixel_location, nearest_indices):
        N = len(regression_targets_S) 
        # N = 5# TODO for test
        prediction = np.zeros_like(regression_targets_S)
        num_targets, lmk_size, dim = regression_targets_S.shape
        self.selected_pixel_index = np.zeros((self.F, 2), dtype=np.uint)
        self.selected_nearest_index = np.zeros((self.F, 2))
        self.selected_pixel_location = np.zeros((self.F, 4), dtype=np.uint)
        self.ci_mat = np.zeros((self.F,N), dtype=np.float32)
        
        # self.corr_mat = np.zeros((self.F, self.P))
        self.corr_mat = np.zeros((self.F, self.P**2))

        self.Fern_threshold = np.random.uniform(0, 1, (self.F, 1))
        
        # start_time = time.time()






        for fi in range(self.F):
            # direction
            # Y_fi = np.random.uniform(-1, 1, size=(lmk_size,dim)) # same as lmk_num, 2
            # Y_fi = np.random.uniform(-1, 1, size=(lmk_size*dim)) # same as lmk_num, 2
            Y_fi = np.random.normal(0, 1, size=(lmk_size*dim)) # same as lmk_num, 2
            # Y_fi_norm = np.linalg.norm(Y_fi, axis=-1)[..., np.newaxis]
            # Y_fi /= Y_fi_norm 
            Y_fi = Y_fi.reshape(-1,1)
            y_proj = regression_targets_S.reshape(N, -1) @ Y_fi
            self.ci_mat[fi, ...] = y_proj.reshape(1,-1)
            # for Ni in range(N):
                # c_i = Y_fi.T @ regression_targets_S[Ni, ...].reshape(-1,1)
                # self.ci_mat[fi, Ni] = c_i
        sample_y_variances = np.var(self.ci_mat, axis=-1)
        
        # calc covariacne between y_proj, P2
        for Fi in range(self.F):
            for Pj in range(self.P):
                self.corr_mat[Fi, Pj] = FirstLevelFern.calc_covariance(self.ci_mat[Fi, :], V[:, Pj])
        # end_time = time.time()
        # print("spent time 1 : %s" %(end_time - start_time))
        # start_time = end_time 

        # corr
                
        
        for mpi in range(self.F):
            max_corr =  -1
            max_pixel_ind = (0,0)
            for mpj in range(self.P):
                for mpk in range(self.P):
                    
                    temp = (P2[mpj, mpj] + P2[mpk, mpk] - 2*P2[mpj, mpk])*sample_y_variances[mpi]
                    if abs(temp) < 1e-10:
                        continue
                    
                    flag = False
                    for pi in range(mpi):
                        if mpj == self.selected_pixel_index[pi, 0] \
                            and mpk == self.selected_pixel_index[pi, 1]:
                            
                            flag = True
                            break
                        elif mpj == self.selected_pixel_index[pi, 1] \
                            and mpk == self.selected_pixel_index[pi, 0]:
                            
                            flag = True 
                            break
                
                    if flag:
                        continue
                    
                    corr = (self.corr_mat[mpi, mpj] - self.corr_mat[mpi, mpk]) / np.sqrt(temp)
                    if abs(corr) > max_corr:
                        max_corr = corr 
                        max_pixel_ind = (mpj, mpk)
            
            self.selected_pixel_index[mpi, :] = max_pixel_ind
            # self.selected_pixel_location[mpi, :2] = pixel_location[max_pixel_ind[0]]
            # self.selected_pixel_location[mpi, 2:] = pixel_location[max_pixel_ind[1]]
            self.selected_nearest_index[mpi, 0] =  nearest_indices[max_pixel_ind[0]]
            self.selected_nearest_index[mpi, 1] =  nearest_indices[max_pixel_ind[1]]
        # end_time = time.time()
        # print("spent time 2 : %s" %(end_time - start_time))
        # start_time = end_time 
        
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
        self.data_in_bins = [[] for i in range(bin_size)]
        for i in range(N):
            index = 0
            for j in range(self.F):
                index_m = self.selected_pixel_index[j, 0]
                index_n = self.selected_pixel_index[j, 1]
                density1 = V[i, index_m]
                density2 = V[i, index_n]
                pixel_diff = density1 - density2 
                if pixel_diff >= self.Fern_threshold[j]:
                    index = int(index + pow(2.0,j))
            self.data_in_bins[index].append( i)

        # prediction for training. yeah-ah
        # end_time = time.time()
        # print("spent time 3 : %s" %(end_time - start_time))
        # start_time = end_time 
        
        prediction = np.zeros_like(regression_targets_S)
        self.bin_params = np.zeros((bin_size, lmk_size, 3), dtype=np.float32)
        for fi in range(bin_size):
            temp = np.zeros((lmk_size, 3))
            sel_bin_size = len(self.data_in_bins[fi])
            for bi in range(sel_bin_size):
                # Si_list, S_i_init_list = data_in_bins[fi][bi] # N x T x 3, N x T x 3
                index = self.data_in_bins[fi][bi] # N x T x 3, N x T x 3
                temp += regression_targets_S[index]
                
            if (sel_bin_size == 0 ):
                    self.bin_params[fi] = temp
                    continue 
            delta_Sb = (1/(1+self.beta / sel_bin_size)) * (temp / sel_bin_size)
            self.bin_params[fi] = delta_Sb
            for bi in range(sel_bin_size):
                index = self.data_in_bins[fi][bi]
                prediction[index] = delta_Sb
        
        # end_time = time.time()
        # print("spent time 4 : %s" %(end_time - start_time))
        # start_time = end_time 
        
        # upate shapes
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
        np.save(os.path.join(save_path, "data_in_bins") ,self.data_in_bins)
        np.save(os.path.join(save_path, "Fern_threshold") ,self.Fern_threshold)
        np.save(os.path.join(save_path, "selected_pixel_index") ,self.selected_pixel_index)
        np.save(os.path.join(save_path, "selected_pixel_location") ,self.selected_pixel_location)
        np.save(os.path.join(save_path, "selected_nearest_index") ,self.selected_nearest_index)
        np.save(os.path.join(save_path, "corr_mat") ,self.corr_mat)
        np.save(os.path.join(save_path, "ci_mat") ,self.ci_mat)


class TwoLevelBoostRegressor:

    """

    """


    def __init__(self, Q = None, nuetral_v = None , T=10, K=300, F=5, beta = 250, P = 400):
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
    
    def refine_3d_lmk(self, lmk2d, **kwargs):
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
        res = get_combine_bar_model(self.neutral_bar, None, self.exprs_bar, None ,expr_weight)
        # B_bar = self.exprs_bar.reshape(-1, v_size*dim).T @ expr_weight
        # B_bar = B_bar.reshape(v_size, dim)
        # res = (new_Rt@ np.concatenate([(self.neutral_bar + B_bar), np.ones((v_size, 1), dtype=np.float32)], axis=-1).T).T
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
            M = train_data_collection[i]['Rt_inv']
            S_init_pose = train_data_collection[i]['S_init']
            S = train_data_collection[i]['S']
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
            


    def train(self, train_data_collection : list, neutral_v : np.ndarray):
        """
            train_data_collection : list of  triandata 
            [Image, Rt, 3d mesh pts, initial 3d mesh pts]
            [I_i, M_i, S_i, S^init_i]
        """
        
        self.weak_regressors = [FirstLevelFern(self.Q, self.K, self.F, self.beta, self.P, name=iid) for iid in range(self.T)]
        N = len(train_data_collection) # training dataset n*m*G*H = 60*9*5*4 = 10,800
        split_data = TwoLevelBoostRegressor.split_data(train_data_collection)
            
        image, M, S_init_pose, S = split_data(0)

        self.lmk_size, dim = S.shape
        self.lmk_shapes = (self.lmk_size, dim)
        S_size = S.size

        #preprocessing
        image_list = []
        M_list = []
        self.S_list = []
        S_init_list = []

        for i in tqdm.trange(N,desc="data split..."):
            image, M, S_init_pose, S = split_data(i)
            image_list.append(image)
            M_list.append(M)
            self.S_list.append(S)
            S_init_list.append(S_init_pose)

        S_list = np.array(self.S_list)
        self.mean_shape = np.zeros_like(self.S_list[0])
        # S_list N x K x 3
        # K x 3
        means = np.mean(S_list, axis=1, keepdims=True)
        centered_shape = S_list - np.mean(S_list, axis=1, keepdims=True)
        scale = np.sqrt(np.mean(np.sum(np.power(centered_shape, 2.0), axis=-1), axis=-1))

        self.mean_shape = np.mean(centered_shape/scale.reshape(-1,1,1), axis=0)
        # np.save(os.path.join(self.save_path, "S"), self.S_list)

        # # initial shapes.
        # self.normalize_transform = []
        # b = self.mean_shape.reshape(-1,1)
        # for i, S in enumerate(self.S_list):
        #     A = np.zeros((3*len(S_list[0]), 9))
        #     for j, raw in enumerate(S):
        #         A[j*3, :3] = raw 
        #         A[j*3+1, 3:6] = raw 
        #         A[j*3+2, 6:] = raw
        #     self.normalize_transform.append(np.linalg.lstsq(A, b)[0])
        current_shapes = np.zeros((len(self.S_list),self.lmk_size, 3), dtype=np.float32)
        for Si, (S_init) in tqdm.tqdm(enumerate(S_init_list), desc="init current shapes."):
            current_shapes[Si, ...] = S_init
        
        regression_targets = np.zeros_like(current_shapes)
            
        
        normlaized_shapes = np.copy(current_shapes)
        normlaized_matrix = np.zeros((len(self.S_list), 3, 3))
        for weak_regressor in tqdm.tqdm(self.weak_regressors, desc="weak regressor train mode."):
            rot_list = []
            rot_inv_list = []
            for i, shape in enumerate(current_shapes):
                image, M, S, cur_shape = image_list[i], M_list[i], S_list[i], current_shapes[i]
                rot = TwoLevelBoostRegressor.normalize_matrix(self.mean_shape, shape)

                normlaized_shapes[i, ...] = ( rot @ shape.T).T

                regression_targets[i, ...] = (rot@ (S - cur_shape).T).T
                rot_list.append(rot)
                rot_inv_list.append(np.linalg.inv(rot_list[i]))

            pred = weak_regressor.train(regression_targets, image_list, current_shapes, rot_list,rot_inv_list, M_list, self.mean_shape)
            for i, p in enumerate(pred):
                p[...] = (rot_inv_list[i] @p.T).T
            current_shapes += pred
            

        self.save_model()


    def predict(self, o, init_num = 15, render= False):
        """
         o : list of images or image 
        """
        def test_rtinv(Rt):
            R = Rt[:,:3]
            Rinv = np.linalg.inv(R)
            invt = - Rinv @ Rt[:, -1, None]
            Rtinv = np.zeros_like(Rt)
            Rtinv[:, :3] = Rinv ; Rtinv[:, -1] = invt.reshape(-1)
            return Rtinv 

        if isinstance(o, list):
            res = []
            for item in o :
                res_item = self._predict(item)
                cv2.imshow("test", res_item)
                cv2.waitKey(100)
                res.append(res_item, init_num)
            return res 
        elif isinstance(o, np.ndarray):
            prev_data = None
            for i in range(100):
                if i== 0:
                    data, new_Rt =self._predict(o, init_num, prev_frame_S=prev_data)
                else : 
                    data, new_Rt =self._predict(o, init_num, prev_frame_S=prev_data)
                Rtinv = test_rtinv(new_Rt)
                if render :
                    if len(o.shape) == 3:
                        h,w,c =(o.shape)
                    else :
                        h,w = o.shape
                    res = proj(self.Q, Rtinv, data)

                    im = vis.draw_circle(res, o, colors=(0,0,255))
                    im = vis.resize_img(im, 1000)
                    vis.set_delay(100)
                    vis.show("result", im)
                    prev_data = (Rtinv[:, :3] @data.T + Rtinv[:, -1, None]).T
            return data
            
        else : 
            raise TypeError("it is not list of images or image(ndarray)")
    

    def similarity_transform(self, src, dest):
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
        ss = [i for i, _ in sorted(enumerate(pose_collection), key = lambda S : np.mean(np.sum(((S[1] - target_pose)**2), axis=-1)))]
        # ss = [i for i, _ in sorted(enumerate(pose_collection), key = lambda S : (np.sum((( (S[1] - np.mean(S[1], axis=0)) - (target_pose - np.mean(target_pose, axis=0)))**2))))]
        return ss[:candidate_num]

    def _predict(self, img, init_num, prev_frame_S = None):
        vis.set_delay(1000)
        ####    for testing 
        self.mean_shape = np.zeros_like(self.S_list[0])
        # S_list N x K x 3
        # K x 3
        self.S_list = np.array(self.S_list)
        means = np.mean(self.S_list, axis=1, keepdims=True)
        centered_shape = self.S_list - np.mean(self.S_list, axis=1, keepdims=True)
        scale = np.sqrt(np.mean(np.sum(np.power(centered_shape, 2.0), axis=-1), axis=-1))

        self.mean_shape = np.mean(centered_shape/scale.reshape(-1,1,1), axis=0)

        #####
        result = np.zeros((self.lmk_size, 3), dtype=np.float32)
        prev_ind = -1
        if prev_frame_S is None : 
            prev_frame_S2d = self.detect_by_dlib(img)
            # im = vis.draw_circle(prev_frame_S2d, img, colors=(0,0,255))
            # im = vis.resize_img(im, 1000)
            # vis.show(title="test",img=im)

            prev_frame_S, new_Rt = self.refine_3d_lmk(prev_frame_S2d, img=img, lmk_idx=lmk_idx)
            prev_frame_S = (new_Rt @ np.concatenate([prev_frame_S, np.ones((len(prev_frame_S),1),dtype=np.float32)], axis=-1).T).T
            # prev_ind = np.random.randint(0, len(self.S_list))
            # prev_frame_S = self.S_list[np.random.randint(0, len(self.S_list))]
        loss = np.inf 
        for i, S in enumerate(self.S_list):
            s_loss = (np.sum((((S - np.mean(S, axis=0)) - (prev_frame_S - np.mean(prev_frame_S, axis=0)))**2)))
            # s_loss = np.sum(((S - prev_frame_S)**2))
            if s_loss < loss: 
                loss = s_loss
                ind = i
        S_r = self.S_list[ind]
        
        
        S_Rp = proj(self.Q,  np.eye(3,4), S_r)
        im = vis.draw_circle(S_Rp, img, colors=(0,0,255))
        res = proj(self.Q,  np.eye(3,4), prev_frame_S)
        im = vis.draw_circle(res, im, colors=(0,255,0))
        im = vis.resize_img(im, 1000)
        vis.show(title="tes2t",img=im)
        trained_img_size = 1000 # for testing
        if len(img.shape) == 3 : 
            intensity_img = cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY)
        else:
            intensity_img = img
        
        #TODO thi is rotation problem ' it is not ortho
        new_Rt = self.similarity_transform(prev_frame_S, S_r) 
        
        prev_frame_S = np.concatenate([prev_frame_S, np.ones((len(prev_frame_S), 1))], axis=-1)
        prev_frame_S = (new_Rt@prev_frame_S.T).T        
        res = proj(self.Q,  np.eye(3,4), prev_frame_S)
        im = vis.draw_circle(res, img, colors=(0,255,0))
        im = vis.draw_circle(S_Rp, im, colors=(0,0,255))
        im = vis.resize_img(im, 1000)
        vis.show(title="S_RandprevS",img=im)
        indices = self.find_most_similar_to_target_pose(pose_collection=self.S_list, \
                                                            target_pose = prev_frame_S, \
                                                            candidate_num=init_num )
        
        # im = np.copy(img)
        # for S in self.S_list:
        #     rr = S
        #     im = vis.draw_circle(proj(self.Q, np.eye(3,4), rr), im, (0,255,255))
        #     vis.show("full_shapes", vis.resize_img(im,1000))

        # im = np.copy(img)
        # for ii in indices:
        #     rr = self.S_list[ii]
        #     im = vis.draw_circle(proj(self.Q, np.eye(3,4), rr), img, (0,255,255))
        #     vis.show("candidate_yello", vis.resize_img(im,1000))


        result = np.zeros_like(prev_frame_S)
        for l in indices:
            cur_pose = np.copy(self.S_list[l])
            im = vis.draw_circle(proj(self.Q, np.eye(3,4), cur_pose), img, (0,0,255))
            # cur_pose = np.concatenate([cur_pose, np.zeros((len(cur_pose), 1))], axis=-1)
            # cur_pose = (new_Rt@cur_pose.T).T
            im = vis.draw_circle(proj(self.Q, new_Rt, cur_pose), im, (0,255,0))

            for weak_regressor in tqdm.tqdm(self.weak_regressors, "weak"):
                pred = weak_regressor.predict(intensity_img, cur_pose, S_prime=prev_frame_S,mean_shape=self.mean_shape, predRt = new_Rt, Q = self.Q)
                cur_pose += pred 
            Rtinv = TwoLevelBoostRegressor.inverse_Rt(new_Rt)
            im = vis.draw_circle(proj(self.Q, Rtinv, cur_pose), im, (255,0,0))
            vis.show("test", vis.resize_img(im,1000))
            result += cur_pose
        result = result / init_num
        return result, new_Rt
                
            

def load_data_train(path):
    meta_path = os.path.join(path, "meta.txt")
    with open(meta_path, 'r') as fp :
        meta = yaml.load(fp, yaml.FullLoader)
        image_ext = meta['image_extension']
        image_root = meta["image_root_location"]
        Q = np.load(os.path.join(path,meta['Q_location']))
        image_location = meta['image_name_location']
        Rt_inv_location = meta['Rt_inv_location']
        S_init_location = meta['S_init_location']
        S_location = meta['S_location']
        root_path = meta['data_root']

    data_root = os.path.join(path, root_path)
    img_names = np.load( os.path.join(data_root, image_location))
    Ss = np.load(os.path.join(data_root, S_location))
    S_inits = np.load(os.path.join(data_root, S_init_location))
    Rt_invs = np.load(os.path.join(data_root, Rt_inv_location))
    res = [] 

    image_dict = {}
    img_names2 = list(set(img_names))
    for name in img_names2:
        img = cv2.imread(os.path.join(image_root, name+image_ext ))
        image_dict[name] = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img)

    for img_name, s, s_init, Rt_inv in tqdm.tqdm(zip(img_names, Ss, S_inits, Rt_invs), "load data"):
        res.append({'img' : image_dict[img_name][0], 'color_img' : image_dict[img_name][1], "S" : s, "S_init" : s_init, "Rt_inv" : Rt_inv})
    return Q, res
        
if __name__ == "__main__":
    Q, data = load_data_train("./cd_test/")
    Q = np.array([[2.29360512e+03, 0.00000000e+00, 2.61489110e+03],
        [0.00000000e+00, 2.29936264e+03, 1.94713585e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
    # for i in range(len(data)):
    #     img = data[i]["color_img"]
    #     S = data[i]["S"]
    #     Rt_inv = data[i]["Rt_inv"]
    #     S_init = data[i]["S_init"]
    #     S_prj = proj(Q, np.eye(3,4,dtype=np.float32), S)
    #     S_i_proj = proj(Q, np.eye(3,4, dtype=np.float32), S_init)
    #     new_img = vis.draw_circle(S_prj, img, colors=(0,0,255))
    #     im = vis.draw_circle(S_i_proj, new_img, colors=(0,255,0))
    #     im = vis.resize_img(im, 1000)
    #     vis.set_delay(100)
    #     vis.show("test", im)

    import igl 
    neutral_v, _ = igl.read_triangle_mesh("data/generic_neutral_mesh.obj")
    neutral_v = neutral_v[lmk_idx, :]
    reg = TwoLevelBoostRegressor(Q = Q, nuetral_v=neutral_v)
    reg.set_save_path("./fern_pretrained_data")
    # reg.train(data, neutral_v)
    reg.load_model("./fern_pretrained_data")
    
    reg.predict(data[0]['color_img'], render = True)