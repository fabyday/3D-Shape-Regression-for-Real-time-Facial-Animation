import numpy as np 



import yaml 
import cv2 
import tqdm

import os 
import visualizer as vis
import scipy.spatial as sp 


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
    def calc_offset_d(Q, Ms, samples, P):
        """
            pts : N x K
            samples : M x K
        """
        # mm = np.array([max_width, max_height, 1], dtype=np.float32).reshape(-1,1)
        R = Ms[:3,:3]
        t = Ms[:, -1, np.newaxis]
        # max_3pts = np.linalg.inv(R)@(np.linalg.inv(Q)@mm - t)
        # a = np.zeros_like(mm)
        # a[-1, -1] = 1
        # min_3pts = np.linalg.inv(R)@(np.linalg.inv(Q)@a - t)

        # centric_pts = np.mean(min_3pts, axis=0, keepdims=True)
        # centered_pts = min_3pts - centric_pts
        
        samples2d = (Q@(R@samples.T+t)).T
        samples2d = samples2d[:, :-1] / samples2d[:, -1, np.newaxis]
        max = np.max(samples, axis=0)
        min = np.min(samples, axis=0)
        # kdtree =sp.KDTree(samples - np.mean(samples, axis=0, keepdims=True))
        # kdtree =sp.KDTree(samples2d)
        # disp = np.random.uniform(min-1, max+1, (P*2, 3))

        disp = np.random.uniform(min-1, max+1, (P, 3))
                     
        kdtree =sp.KDTree(samples)
        # disp = np.random.normal(np.mean( samples ,axis=0, keepdims=True), 2,size= (10000, 3))
        # disp[:, -1] = 1
        _, nearest_index = kdtree.query(disp)
        # disp2d = (Q@(R@disp.T+t)).T
        disp2d = (Q@(R@disp.T+t)).T
        disp2d = disp2d[:, :-1] / disp[:, -1, np.newaxis]
        # _, nearest_index = kdtree.query(disp2d)
        print(set(nearest_index))
        print(disp2d)
        print(samples2d)

        import matplotlib.pyplot as plt 

        # plt.scatter(disp2d[:, 0], disp2d[:, 1], color='r')
        # plt.scatter(samples2d[:, 0], samples2d[:, 1], color='b')
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(disp[:, 0], disp[:, 1], disp[:, 2], marker="x")
        # ax.scatter(samples[:,0], samples[:,1], samples[:,2], marker="o")
        # plt.show()
        resdisp = np.zeros_like(disp)
        
        for i, nearest_idx in enumerate(nearest_index):
            disp[i, :] -= samples[nearest_idx, :] 
        
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
        for ii, ni in enumerate(nearset_index):
            # p[ni, :] = S_init_pose[ni] + disp[ii]
            p[ii, :] = S_init_pose[ni] + disp[ii]
        pts_2d = proj(Q, M, p)


        # test for draw
        # im = np.copy(Image)
        # import visualizer
        # im = visualizer.draw_circle(pts_2d, im)
        # im = visualizer.resize_img(im, 1000)
        # cv2.imshow("test", im)
        if len(Image.shape) >= 3 :
            img_intensity = cv2.cvtColor(Image,  cv2.COLOR_BGR2GRAY)
        else : 
            img_intensity = Image
        # img_intensity = FirstLevelFern.convert_RGB_to_intensity(Image)
        # img_intensity = FirstLevelFern.convert_RGB_to_intensity(Image)
        loc = pts_2d.astype(np.uint)
          

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
    def convert_RGB_to_luminance(img):
        # see also https://stackoverflow.com/questions/56198778/what-is-the-efficient-way-to-calculate-human-eye-contrast-difference-for-rgb-val/56237498#56237498
        # BGR
        return img[:,:, 0] * 0.0722 + img[:,:, 0] * 0.7152 + img[:,:, 0] * 0.2125


    def train(self, current_shapes, image_list, M_list, S_list, neutral_v):
        lmk_size, dim = S_list[0].shape
        regression_targets = np.zeros_like(current_shapes, dtype=np.float32) # N x lmk_size x 2
        target_num, lmk_size, dim = regression_targets.shape 

        for i in range(target_num):
            image, M, S, S_init_pose = image_list[i], M_list[i], S_list[i], np.squeeze(current_shapes[i, ...])
            regression_targets[i, ...] = S - S_init_pose

        N = len(current_shapes) # TODO disable it when real-trainig. 
        # N = 5 # for testing. remove it when you train model. this is toy model
    
        self.V_list = [] # N x P x 1
        self.pixel_loc_list = [] # N x P x 2
        self.nearest_index_list = [] # N x P x 1

        image, M, S, S_init_pose = image_list[i], M_list[i], S_list[i], np.squeeze(current_shapes[0, ...])
        self.nearset_index, self.disp = FirstLevelFern.calc_offset_d(self.Q, M, neutral_v, self.P)

        for i in range(N): # TODO
            image, M, S, S_init_pose = image_list[i], M_list[i], S_list[i], np.squeeze(current_shapes[i, ...])
            V_i, pixel_loc, nearest_index, _  = FirstLevelFern.calc_appearance_vector(image,self.Q, M, S_init_pose, self.P, self.disp ,self.nearset_index)
            # self.nearest_index_list.append(nearest_index)
            self.pixel_loc_list.append(pixel_loc)
            self.V_list.append(V_i)

        
        # intensity
        self.V = np.array(self.V_list, dtype=np.float32).reshape(N, self.P) # N x P x 1
        self.V = self.V.T # P x N 
        self.pixel_location = np.array(self.pixel_loc_list)
        # self.nearest_index = np.array(self.nearest_index_list)
        
        self.P2 = np.zeros((self.P, self.P))
        #calc P2 
        for i in range(self.P):
            for j in range(i, self.P):
                correlation = FirstLevelFern.calc_covariance(self.V[i, :], self.V[j, :])
                self.P2[i, j] = correlation
                self.P2[j, i] = correlation
            

        prediction = np.zeros_like(regression_targets)
        # for ki in tqdm.trange(self.K): #second level regression
        for fern_regressor in tqdm.tqdm(self.ferns, desc="train fern"):
            pred = fern_regressor.train(regression_targets, self.P2, self.V, self.pixel_location, self.nearset_index)
            prediction += pred 
            regression_targets -= pred


        return prediction
            

    def predict(self, img, cur_S, S_prime, Q):
        
        intensity_vectors, loc, nearset_index, disp  = FirstLevelFern.calc_appearance_vector(img, Q, np.eye(3,4), cur_S, 400, self.disp, self.nearest_index_list)
        for fern_reg in tqdm.tqdm(self.ferns, "test fern"):
            cur_S += fern_reg.pred(img, cur_S, Q, intensity_vectors)
        return cur_S

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
    def pred(self, img, cur_S, Q, intensity_vector):
        index = 0
        self.selected_nearest_index = self.selected_nearest_index.astype(np.uint)
        for f in ( range( self.F)):
            nearest_lmk_idx_1 = self.selected_nearest_index[f, 0]
            nearest_lmk_idx_2 = self.selected_nearest_index[f, 1]
            x = self.selected_pixel_location[f, 0]
            y = self.selected_pixel_location[f, 1]


            intensity_1 = intensity_vector[nearest_lmk_idx_1]
            intensity_2 = intensity_vector[nearest_lmk_idx_2]

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

            if(intensity_1 - intensity_2 >= self.Fern_threshold[f]):
                index = index +int(pow(2, f))
        return self.bin_params[index]



    def train(self, regression_targets,P2, V, pixel_location, nearest_indices):
        N = len(regression_targets) 
        # N = 5# TODO for test
        prediction = np.zeros_like(regression_targets)
        num_targets, lmk_size, dim = regression_targets.shape
        self.selected_pixel_index = np.zeros((self.F, 2), dtype=np.uint)
        self.selected_nearest_index = np.zeros((self.F, 2))
        self.selected_pixel_location = np.zeros((self.F, 4), dtype=np.uint)
        self.ci_mat = np.zeros((self.F,N), dtype=np.float32)
        
        self.corr_mat = np.zeros((self.F, self.P))

        self.Fern_threshold = np.random.uniform(0, 1, (self.F, 1))
        for fi in range(self.F):
            # direction
            Y_fi = np.random.uniform(-1, 1, size=(lmk_size, dim)) # same as lmk_num, 2
            Y_fi_norm = np.linalg.norm(Y_fi, axis=-1)[..., np.newaxis]
            Y_fi /= Y_fi_norm 
            Y_fi = Y_fi.reshape(-1,1)
            for Ni in range(N):
                c_i = Y_fi.T @ regression_targets[Ni, ...].reshape(-1,1)
                self.ci_mat[fi, Ni] = c_i
        
        for Fi in range(self.F):
            for Pj in range(self.P):
                self.corr_mat[Fi, Pj] = FirstLevelFern.calc_covariance(self.ci_mat[Fi, :], V[Pj, :])
        
        max_corr =  -1
        max_pixel_ind = (0,0)
        for mpi in range(self.F): 
            for mpj in range(self.P):
                for mpk in range(self.P):
                    
                    temp = P2[mpj, mpj] + P2[mpk, mpk] - 2*P2[mpj, mpk]
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
                    
                    corr = (self.corr_mat[mpi, mpj] - self.corr_mat[mpi, mpk]) / temp
                    if abs(corr) > max_corr:
                        max_corr = temp 
                        max_pixel_ind = (mpj, mpk)
            
            self.selected_pixel_index[mpi, :] = max_pixel_ind
            self.selected_pixel_location[mpi, :2] = pixel_location[mpi][max_pixel_ind[0]]
            self.selected_pixel_location[mpi, 2:] = pixel_location[mpi][max_pixel_ind[1]]
            self.selected_nearest_index[mpi, 0] =  nearest_indices[max_pixel_ind[0]]
            self.selected_nearest_index[mpi, 1] =  nearest_indices[max_pixel_ind[1]]

        max_diff = -1 # make threshold
        for ii in range(self.F):
            for intensity_j, intensity_k in zip(V[self.selected_pixel_index[ii, 0], :], V[self.selected_pixel_index[ii, 1], :]) : 
                temp = intensity_j - intensity_k 
                if abs(temp) > max_diff:
                    max_diff = abs(temp)
            self.Fern_threshold[ii] = np.random.uniform(-0.2*max_diff, 0.2*max_diff) 
        

        bin_size = 2**self.F
        self.data_in_bins = [[] for i in range(bin_size)]
        for i in range(N):
            index = 0
            for j in range(self.F):
                density1 = V[self.selected_pixel_index[j, 0]][i]
                density2 = V [self.selected_pixel_index[j, 1]][i]
                if density1 - density2 >= self.Fern_threshold[j]:
                    index = int(index + pow(2.0,j))
            self.data_in_bins[index].append( i)

        # prediction for training. yeah-ah
            
        prediction = np.zeros_like(regression_targets)
        self.bin_params = np.zeros((bin_size, lmk_size, 3), dtype=np.float32)
        for fi in range(bin_size):
            temp = np.zeros((lmk_size, 3))
            sel_bin_size = len(self.data_in_bins[fi])
            for bi in range(sel_bin_size):
                # Si_list, S_i_init_list = data_in_bins[fi][bi] # N x T x 3, N x T x 3
                index = self.data_in_bins[fi][bi] # N x T x 3, N x T x 3
                temp += regression_targets[index]
                
            if (sel_bin_size == 0 ):
                    self.bin_params[fi] = temp
                    continue 
            delta_Sb = (1/(1+self.beta / sel_bin_size)) * (temp / sel_bin_size)
            self.bin_params[fi] = delta_Sb
            for bi in range(sel_bin_size):
                index = self.data_in_bins[fi][bi]
                prediction[index] = delta_Sb
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
        meta_dict = {"T" : self.T, "K" : self.K, "F" : self.F, "Q" : "Q.npy", "P" : self.P, "beta" : self.beta, "lmk_size": self.lmk_size, "S" : "S.npy"}
        with open(os.path.join(self.save_path,"meta.txt"), "w") as fp :
            yaml.dump(meta_dict, fp)
        for weak_reg in tqdm.tqdm(self.weak_regressors, "save weak regressors"):
            weak_reg.save(self.save_path)

    def load_model(self, path):
        with open(os.path.join(path, "meta.txt"), 'r') as fp :
            meta = yaml.load(fp, yaml.FullLoader)
            self.T = meta['T']
            self.K = meta['K']
            self.F = meta['F']
            self.Q = np.load(os.path.join(self.save_path, meta['Q']))
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
        # np.save(os.path.join(self.save_path, "S"), self.S_list)

        # initial shapes.
        current_shapes = np.zeros((len(self.S_list),self.lmk_size, 3), dtype=np.float32)
        for Si, (S_init) in tqdm.tqdm(enumerate(S_init_list), desc="init current shapes."):
            current_shapes[Si, ...] =   S_init
        
        for weak_regressor in tqdm.tqdm(self.weak_regressors, desc="weak regressor train mode."):
            pred = weak_regressor.train(current_shapes, image_list, M_list, self.S_list, neutral_v)
            current_shapes += pred
            

        self.save_model()


    def predict(self, o, init_num = 15, render= False):
        """
         o : list of images or image 
        """
        if isinstance(o, list):
            res = []
            for item in o :
                res_item = self._predict(item)
                cv2.imshow("test", res_item)
                cv2.waitKey(100)
                res.append(res_item, init_num)
            return res 
        elif isinstance(o, np.ndarray):
            data, new_Rt =self._predict(o, init_num)
            Rtinv = np.linalg.inv(new_Rt[:3,:3])
            tinv = -new_Rt[:, -1, None]
            Rtinv = np.concatenate([Rtinv, tinv], -1)
            if render :
                h,w,c =(o.shape)
                res = proj(self.Q, Rtinv, data)
                im = vis.draw_circle(res, o, colors=(0,0,255))
                im = vis.resize_img(im, 1000)
                vis.show("test", im)
                vis.set_delay(0)
            return data
            
        else : 
            raise TypeError("it is not list of images or image(ndarray)")
    

    def similarity_transform(self, S, S_prime):
        """
            Ruturn Rt that S_prime to S
        """
        S_prime_H = np.concatenate([S_prime, np.ones((len(S_prime), 1), dtype = np.float32)], axis=1)

        A = np.zeros((3*len(S_prime), 12), dtype=np.float32)

        for i in range(len(S_prime_H)):
            A[i*3   , :4]  = S_prime_H[i]
            A[i*3+1 , 4:8]  = S_prime_H[i]
            A[i*3+2 , 8:]  = S_prime_H[i]

        flat_S = S.reshape(-1, 1)
        
        res = np.linalg.lstsq(A, flat_S)
        Rt = res[0].reshape(3,4)
        return Rt

        

    def find_most_similar_to_target_pose(self, pose_collection, target_pose, candidate_num = 20):
        
        # TODO for testing.
        return np.random.randint(0, len(pose_collection))

    def _predict(self, img, init_num, prev_frame_S = None):
        result = np.zeros((self.lmk_size, 3), dtype=np.float32)
        trained_img_size = 1000 # for testing
        intensity_img = cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY)

        if prev_frame_S != None : 
            index = self.find_most_similar_to_target_pose(pose_collection=self.S_list, \
                                                            target_pose = prev_frame_S, \
                                                            candidate_num=init_num )
        else : 
            index = np.random.randint(0, trained_img_size) # init lmk index


        for l in range(init_num):
            cur_pose = self.S_list[index]

            new_Rt = self.similarity_transform( self.nuetral_v, cur_pose)
            cur_pose = np.concatenate([cur_pose, np.zeros((len(cur_pose), 1))], axis=-1)
            cur_pose = (new_Rt@cur_pose.T).T
            for weak_regressor in self.weak_regressors:
                pred = weak_regressor.predict(intensity_img, cur_pose, S_prime=cur_pose, Q = self.Q)
                cur_pose += pred 
            result = cur_pose / init_num
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
        image_dict[name] = img
    for img_name, s, s_init, Rt_inv in zip(img_names, Ss, S_inits, Rt_invs):
        res.append({'img' : image_dict[img_name], "S" : s, "S_init" : s_init, "Rt_inv" : Rt_inv})
    return Q, res
        
if __name__ == "__main__":
    Q, data = load_data_train("./cd_test/")
    import igl 
    neutral_v, _ = igl.read_triangle_mesh("data/generic_neutral_mesh.obj")
    neutral_v = neutral_v[lmk_idx, :]
    reg = TwoLevelBoostRegressor(Q = Q, nuetral_v=neutral_v)
    reg.set_save_path("./fern_pretrained_data")
    # reg.train(data, neutral_v)
    reg.load_model("./fern_pretrained_data")
    
    reg.predict(data[0]['img'], render = True)