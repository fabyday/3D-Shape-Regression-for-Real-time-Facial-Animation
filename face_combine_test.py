import igl 

import preprop as pp 
import numpy as np 

np.random.normal(0, 1.0)


p = pp.PreProp("lmks", "images", "prep_data")
p.build(4) 

def test(p : pp.PreProp, low :float, high:float, name :str):
    meshes = p.meshes
    ids = np.array( [m[0] for m  in meshes])
    neutral_id = np.mean(ids, axis=0, keepdims=True)
    r,c = meshes[0][0].shape
    ids = ids.reshape(-1, 1, r, c)
    exps = np.array([m[1:] for m  in meshes])
    id_size, exp_size, exp_r, exp_c = exps.shape
    neutral_exp = np.mean(exps.reshape(-1, exp_r, exp_c), axis=0)
    exps -= neutral_exp 
    id_size, expr_size, _, _ = exps.shape
    exps = exps.reshape(id_size*exp_size, exp_r*exp_c).T
    ids = ids.reshape(-1, r*c).T
    id_weight = np.random.uniform(0, 1, size = (id_size,1))
    exp_weight = np.random.uniform(0, 1, size = (id_size*expr_size, 1))
    if low ==0.0 and high == 0.0 : 
        id_weight = np.zeros_like(id_weight)
        exp_weight = np.zeros_like(exp_weight)
            
    data_mean = neutral_id + neutral_exp
    # result = data_mean + (exps@exp_weight + ids@id_weight).reshape(1, r,c)
    result = data_mean
    igl.write_triangle_mesh(name, result.squeeze(), p.ref_f)

import os 
if not os.path.exists("face_conbine_test_folder"):
    os.makedirs("face_conbine_test_folder")
test(p, 0, 0 , "face_conbine_test_folder/mean.obj")
test(p, 0, 1 , "face_conbine_test_folder/test1.obj")
test(p, 0, 1 , "face_conbine_test_folder/test2.obj")