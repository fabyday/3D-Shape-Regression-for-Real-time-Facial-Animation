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
    eyebrow = raw_meta['meta']['ict_landmark_index']['eyebrow']
    nose = raw_meta['meta']['ict_landmark_index']['nose']
    return full_index, eye, contour, mouse, eyebrow, nose

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



def load_extracted_lmk_meta(name):
    raw_meta = get_yaml_data(name)
    file_ext = raw_meta['meta']['file_ext']
    image_root = raw_meta['meta']['images_root']
    images = raw_meta['meta']['images_name']
    return images, image_root, file_ext
