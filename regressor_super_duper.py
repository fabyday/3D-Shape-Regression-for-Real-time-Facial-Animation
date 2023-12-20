import numpy as np 



import tqdm

import os 

import scipy.spatial as sp 


class TwoLevelBoostRegressor:

    """

    """


    def __init__(self, Q, T=10, K=300, F=5, beta = 250, P = 400):
        """
            T : The number of first level regression
            K : The number of second level regression
            F : size of bin in weak regressor 
            Q : camera intrinsic matrix (3x3)

        """
        self.T = T 
        self.K = K 
        self.F = F 
        self.Q = Q
        self.P = P
        self.beta = beta
    
    @staticmethod
    def generate_offset_d(P = 400):
        loc = np.zeros((1,3))
        samples = np.random.normal(loc=loc, size=(P, 3))
        return samples


    
    @staticmethod
    def calc_offset_d(pts, samples):
        """
            pts : N x K
            samples : M x K
        """
        
        centric_pts = np.mean(pts, axis=0, keepdims=True)
        centered_pts = pts - centric_pts
        kdtree =sp.KDTree(centered_pts)
        _, nearest_index = kdtree.query(samples)
        disp = np.zeros_like(samples)
        
        for i, nearest_idx in enumerate(nearest_index):
            disp[i, :] = samples[nearest_idx, :] 
        
        return nearest_index, disp
    
    @staticmethod
    def calc_appearance_vector(Image, Q, M, S_init_pose, disp):
        
        nearset_index, disp = TwoLevelBoostRegressor.calc_offset_d(S_init_pose, disp)
        # Vi := (intensity : ndarray  Nx2,  P_points Nx3)
        R = M [:3, :3]
        t = M [:, -1]
        pts_2d_h = (Q @ (M @ S_init_pose.T + t)).T
        pts_2d = pts_2d_h[:, :-1] / pts_2d_h[:, -1]
        intensity_vectors = pts_2d  # N x 2
        

    def train(self, train_data_collection : list):
        """
            train_data_collection : list of  triandata 
            [Image, Rt, 3d mesh pts, initial 3d mesh pts]
            [I_i, M_i, S_i, S^init_i]
        """
        

        N = 10800 # training dataset n*m*G*H = 60*9*5*4 = 10,800
        def split_data(i):
            image = train_data_collection[i]['img']
            M = train_data_collection[i]['Rt_inv']
            S_init_pose = train_data_collection[i]['S_init']
            S = train_data_collection[i]['S']
            return image, M, S_init_pose, S
        image, M, S_init_pose, S = split_data( i)
        S_size = S.size
        for ti in tqdm.trange(self.T): # level 1 

            disp = TwoLevelBoostRegressor.generate_offset_d()
            V_list = []
            for i in tqdm.trange(N):
                image, M, S_init_pose, S = split_data( i)
                V_i = TwoLevelBoostRegressor.calc_appearance_vector(image, M, S_init_pose, disp)
                V_list.append(V_i)


            for ki in tqdm.trange(self.K):
                F_index_pair = []
                for fi in tqdm.trange(self.F):

                    Y_fi = np.random.normal((0,0), size=(1,S_size))
                    ci_list = []
                    for i in tqdm.trange(1, N):
                        image_i, M_i, S_i_init_pose, S_i = split_data(i)
                        delta_S_i = S_i - S_i_init_pose
                        c_i = Y_fi.T @ delta_S_i
                        ci_list.append(c_i)
                
                data_in_bins = [[] for i in range(2**self.F)]
                for i in tqdm.trange(N):
                    pass 

                for fi in tqdm.trange(2**self.F):
                    Si_list, S_i_init_list = data_in_bins[fi] # N x T x 3, N x T x 3
                    sum_Ss = np.sum(np.array(Si_list) - np.array(S_i_init_list), axis=0) # Tx3
                    omega_Si = len(Si_list)
                    delta_Sb = (1/(1+self.beta/omega_Si)) * (sum_Ss/omega_Si)





    def predict(self, o):
        """
         o : list of images or image 
        """
        if isinstance(o, list):
            pass 
        elif isinstance(o, np.ndarray):
            pass
        else : 
            raise TypeError("it is not list of images or image(ndarray)")
        





        
if __name__ == "__main__":
    reg = TwoLevelBoostRegressor(Q = np.identity(3))
    reg.train()