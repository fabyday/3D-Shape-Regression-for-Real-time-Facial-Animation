#deformation transfer




import trimesh 
import igl 
import numpy as np 
import glob, os 
import os.path as osp 





lmks = [1225,1888,1052,367,1719,1722,2199,1447,966,3661,4390,3927,3924,2608,3272,4088,3443,268,493,1914,2044,1401,3615,4240,4114,2734,2509,978,4527,4942,4857,1140,2075,1147,4269,3360,1507,1542,1537,1528,1518,1511,3742,3751,3756,3721,3725,3732,5708,5695,2081,0,4275,6200,6213,6346,6461,5518,5957,5841,5702,5711,5533,6216,6207,6470,5517,5966]
lmks = [i for i in range(6705)]


def build_expr(src_ref, tgt_refs, src_expr, lmks):
    (name, src_ref_obj) = src_ref
    lmks = np.array(lmks)
    res = []
    for tgt_name, tgt in tgt_refs:
        tgt_expr = []
        for expr_name, expr in src_expr:
            lmks_v = expr.vertices[lmks]
            # print(lmks_v)
            ll = trimesh.registration.nricp_sumner(src_ref_obj, tgt, lmks, lmks_v)
            # print(ll)
            tgt_expr.append((expr_name, ll))
        res.append((tgt_name, tgt_expr))
    return res




def load_tgt_identities(path):
    import re

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

    identity_path = glob.glob(osp.join(path, "identity**.obj"))
    identity_path.sort(key= natural_keys)
    tgt_identity = []
    for pp in identity_path:
        
        iden = trimesh.load_mesh(pp)
        tgt_identity.append((osp.basename(pp), iden))
    return tgt_identity

def load_src_identity_expr(path):
    ref = trimesh.load_mesh(osp.join(path, "generic_neutral_mesh.obj"))
    expr_path = glob.glob(osp.join(path, "shapes","**.obj"))
    exprs = []
    for pp in expr_path:
        expr = trimesh.load_mesh(pp)
        exprs.append((osp.basename(pp),expr))
    return ("generic_neutral_mesh.obj", ref), exprs


def save_all(path, src_ref, tgt_identities, src_exprs, tgts_exprs):
    if not osp.exists(path):
        os.makedirs(path)
    name, src_obj = src_ref
    igl.write_triangle_mesh(osp.join(path, name), src_obj.vertices, src_obj.faces)
    expr_path = osp.join(path, "shapes")
    src_path = osp.join(expr_path, osp.splitext(name)[0])
    
    if not osp.exists(src_path):
        os.makedirs(src_path)
    for name, o in src_exprs:
        igl.write_triangle_mesh(osp.join(src_path, name), o.vertices, o.faces)
    
    for name, obj in tgt_identities:
        igl.write_triangle_mesh(osp.join(path, name), obj.vertices, obj.faces)


    for tgt_name, tgt_exprs_tup in tgts_exprs:
        tgt_expr_path = osp.join(expr_path, osp.splitext(tgt_name)[0])
        if not osp.exists(tgt_expr_path):
            os.makedirs(tgt_expr_path)
        for name, v in tgt_exprs_tup:
            igl.write_triangle_mesh(osp.join(tgt_expr_path, name), v, src_obj.faces)

def main():
    path = "data"
    spth = "prep_data"
    print("load src...")
    src_ref, src_exprs = load_src_identity_expr("data")
    print("load tgt...")
    tgt_identities = load_tgt_identities("data")
    print("build tgt...")
    tgt_expr = build_expr(src_ref, tgt_identities, src_exprs, lmks)
    print("save tgt...")
    save_all(spth, src_ref, tgt_identities, src_exprs, tgt_expr)





if __name__ == "__main__":
    main()
