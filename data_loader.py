import yaml 


import os.path as osp
name = "test"
def get_yaml_data(name):
    with open(name, 'r') as fp:
        raw_meta = yaml.load(fp, yaml.FullLoader)
    return raw_meta
    


def load_ict_landmark(name):
    raw_meta = get_yaml_data(name)
    full_index = raw_meta['meta']['ict_landmark_index']['full_index']
    eye = raw_meta['meta']['ict_landmark_index']['eyes']
    contour = raw_meta['meta']['ict_landmark_index']['contour']
    mouse = raw_meta['meta']['ict_landmark_index']['mouse']
    return full_index, eye, contour, mouse

def load_image_meta(name):
    """
    return meta data:
    file ext
    """
    raw_meta = get_yaml_data(name)
    keys = raw_meta["meta"]["images_name"].keys()
    meta = raw_meta["meta"]["images_name"]
    ext =  raw_meta["meta"]["file_ext"]

    return meta, ext

