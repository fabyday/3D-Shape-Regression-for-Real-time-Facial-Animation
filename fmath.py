import numpy as np 


def normalize(x):
    mean = np.mean(x, 0)
    std = np.std(x, 0)
    normalized_x = (x - mean)/std
    return normalized_x, mean, std

def add_Rt_to_mesh(Rt, v):
    new_Rt = Rt[:3,:3]
    new_trans = Rt[:, -1, np.newaxis]
    return (new_Rt@v.T + new_trans).T

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


############## Mesh 


def get_combine_model(neutral, ids=None, expr=None, w_i=None, w_e=None):
    
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

def get_bars(neutral, ids, exprs, sel_lmk_idx):
    neutral_bar = neutral[sel_lmk_idx, :]
    ids_bar = ids[:, sel_lmk_idx, :]
    expr_bar = exprs[:, sel_lmk_idx, :]
    return neutral_bar, ids_bar, expr_bar

def add_Rt_to_pts(Q, Rt, x):
    R = Rt[:3,:3]
    t = Rt[:, -1, None]
    xt = x.T
    Rx = R @ xt 
    Rxt = Rx+t
    pj_Rxt = Q @ Rxt
    res = pj_Rxt/pj_Rxt[-1, :]
    return res[:2, :].T
    

def default_cost_function(Q, Rt, neutral_bar, ids_bar, exprs_bar, id_weight, expr_weight, y):
    # x := Q + Rt + expr weight

    blended_pose = get_combine_bar_model(neutral_bar , ids_bar, exprs_bar, id_weight, expr_weight)
    
    gen = add_Rt_to_pts(Q, Rt, blended_pose)
    z = gen - y
    new_z = z.reshape(-1, 1)
    new_z = new_z.T @ new_z
    return new_z


def default_cost_function_mt(args):
    Q, Rt, neutral_bar, ids_bar, exprs_bar, id_weight, expr_weight, lmk_2d =args
    return default_cost_function(Q, Rt, neutral_bar, ids_bar, exprs_bar, id_weight, expr_weight, lmk_2d) 