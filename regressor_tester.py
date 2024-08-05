import regressor_super_duper as reg


import cv2, igl
import numpy as np 
class RegTester(reg.TwoLevelBoostRegressor):
   
    


    def test_predict(self, img, init_lmk, Rt, clmk_idx=None, Q=None, mean_shape = None ):
        intensity_img = cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        inv_RR = Rt
        cur_pose = np.copy(init_lmk)
        for weak_regressor in self.weak_regressors:
            # cur_pose = weak_regressor.predict(intensity_img, cur_pose, S_prime=prev_frame_S,mean_shape=self.mean_shape, predRt = InvRt, Q = self.Q)
            cur_pose += weak_regressor.predict(intensity_img, cur_pose, S_prime=None,mean_shape=mean_shape, predRt = inv_RR[:3,:], Q = self.Q)
            # pred = weak_regressor.predict(intensity_img, cur_pose, S_prime=prev_frame_S,mean_shape=self.mean_shape, predRt = combined_Rt[:3,:], Q = self.Q)
            # cur_pose += pred
        return cur_pose

Q, data, Ss, S_original, Rt_invs, resize_ratio = reg.load_data_train("./kinect_test_dataset/")
Q = reg.get_kinect_Q()

neutral_v, _ = igl.read_triangle_mesh("data/generic_neutral_mesh.obj")


reg_test = RegTester(Q = Q,beta=250, Ss=Ss, S_original = S_original, neutral_v=neutral_v, data=data,P=50)

reg_test.set_save_path("./fern_pretrained_data")
reg_test.load_model("./fern_pretrained_data")
split_ = reg_test.split_data(data)

print(S_original)
mean_shape = np.mean(Ss, axis=0)
for i in range(len(data)):
    image, M, S_init_pose, S_index = split_(i)
    Rt_inv100 = Rt_invs[M]
    S_100 = Ss[S_init_pose]
    pred = reg_test.test_predict(image, S_100, Rt_inv100, None, Q=Q, mean_shape = mean_shape)

    h,*_ = image.shape
    image = cv2.cvtColor(image,  cv2.COLOR_GRAY2BGR)

    S_Gt = Ss[M]

    Gtpp = reg.proj(Q, Rt_inv100, S_Gt) #gt
    pp = reg.proj(Q, Rt_inv100, pred) # pred
    S100pp = reg.proj(Q, Rt_inv100, S_100) # real
    Gtpp = reg.geo.convert_to_cv_image_coord(Gtpp, h)
    pp = reg.geo.convert_to_cv_image_coord(pp, h)
    S100pp = reg.geo.convert_to_cv_image_coord(S100pp, h)

    # Green Predict Start Point
    # Blue : pred 
    # Red Gt
    image = reg.vis.draw_circle(S100pp, image, (0,255, 0), 2, thickness=2)
    image = reg.vis.draw_circle(Gtpp, image, (0,0,255), 2, thickness=2)
    image = reg.vis.draw_circle(pp, image, (255,0,0), 2, thickness=2)
    reg.vis.set_delay(0)

    reg.vis.show("test", image)
