import geo_func 





import numpy as np 

truth = np.random.rand(4,2)
rad = np.deg2rad(45)
truth_rot = np.array([np.cos(rad), -np.sin(rad), np.sin(rad), np.cos(rad)]).reshape(2,2)
scale = 3.5
test_input = scale*truth_rot@truth.T
test_input = test_input.T

def test_similarity(A, B):
    rot_t, scale_t = geo_func.similarity_transform(A,B)
    print(rot_t == truth_rot)
    print(scale_t == scale)
    print(scale, " ", scale_t)
    print(truth_rot, " \n", rot_t)

    print(A)
    print((scale_t*rot_t@A.T).T)
    print("=======")
    print(B)
    print((scale_t*rot_t@B.T ).T)
def test_similarity2(A, B):
    rot_t = geo_func.similarity_transform2(A,B)
    print(rot_t == truth_rot)
    print(rot_t == scale*truth_rot)
    print("truyt rot", scale*truth_rot, " gen rot \n", rot_t)

    print("A\n", A)
    print("RA\n",(rot_t@A.T).T)
    print("=======")
    print("B\n", B)



# test_similarity(truth, test_input)
test_similarity2(truth, test_input)