import yaml 
import numpy as np
import cv2
import os.path as osp
def get_yaml_data(name):
    with open(name, 'r') as fp:
        raw_meta = yaml.load(fp, yaml.FullLoader)
    return raw_meta
    

def load_extracted_lmk_meta(name):
    raw_meta = get_yaml_data(name)
    file_ext = raw_meta['meta']['file_ext']
    image_root = raw_meta['meta']['images_root']
    images = raw_meta['meta']['images_name']
    return images, image_root, file_ext

class PreProp:
    def __init__(self, meta_dir, mesh_dir):
        
        self.proj = np.identity(4)
        self.rigid_trans = np.identity(4)
        self.meta_location = meta_dir 

        self.img_meta, self.img_root, self.img_file_ext,  = load_extracted_lmk_meta(self.meta_location)

        self.mesh_dir = mesh_dir



    def build(self, cutter = None):
        imgs = self.load_data()

            



    def load_data(self):
        extension = [".jpeg", ".png", ".jpg"]

        def read_lmk_meta(path):
            lmk = []
            with open(path, "r") as fp: 
                while True:
                    ss = fp.readline()
                    if not ss:
                        break
                    ss = ss.rstrip("\n")
                    x, y = ss.split(" ")
                    lmk.append([float(x),float(y)])
            return lmk

        self.img_meta.keys()


        self.img_and_info = dict()
        self.img_list = []
        for key in self.img_meta.keys():
            category = self.img_and_info.get(key, None)
            meta_item = self.img_meta[key] # into category
            if category == None :
                self.img_and_info[key] = []
                category = self.img_and_info[key]
            
            for meta_data in meta_item:
                meta_data['landmark']
                name = meta_data['name']
                lmk_data = read_lmk_meta(meta_data['landmark'])
                img_data = cv2.imread(osp.join(self.img_root, name+self.img_file_ext))
                img_data = {"name" : name, "lmk_data" : lmk_data, "img_data": img_data}
                self.img_list.append(img_data)
                category.append(img_data)

        return self.img_list
    
p = PreProp("./meta.yaml", None)
p.build()

def draw_circle(v, img, colors = (1.0,0.0,0.0)):
    for vv in v:
        cv2.circle(img, center=vv.astype(int), radius=10, color=colors, thickness=2)

for img in p.img_list:
    h,w,_ = img['img_data'].shape
    default_img = img['img_data']
    draw_circle(np.array(img['lmk_data']), default_img) 
    print(np.array(img['lmk_data']))
    resized_img = cv2.resize(default_img, [int(w//5), int(h//5)])
    cv2.imshow("test", resized_img)
    cv2.waitKey(0)