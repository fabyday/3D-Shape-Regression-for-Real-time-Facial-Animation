import os  

import os.path as osp 
import yaml 
import typing 
class BaseFaceMeta:



    class ComponentMetaItem():
        def __init__(self, name = ""):
            self.m_name = name
            self.m_data = {}
            self.m_indice_list = []
        @property
        def name(self):
            return self.m_name

        
        def deserialize(self, name, meta):
            self.m_name = name
            if isinstance(meta, list ) : # is means end-points
                self.m_indice_list = meta 
                return 
        
            keys = list(meta.keys())
            
            if meta.get("full_index", False):
                self.m_indice_list = meta.get("full_index")

            #if full_index exists
            keys = list(filter(lambda x : x != "full_index", keys))
            for key in keys:
                item = BaseFaceMeta.ComponentMetaItem()
                item.deserialize(key, meta[key])
                self.m_data[key] = item

      
        @property
        def keys(self):
            return self.m_data.keys() 
        
        def get_indice_list(self): 
            """
                append child list
            """
            return self.m_indice_list

        def __getitem__(self, key : str): 
            keys = key.split(".")
            if len(keys) >= 2:
                key, keys = keys 
                
            return self.m_data[key][keys]

    DEFAULT_META_NAME = "meta.yaml"
    def __init__(self):
        self.m_raw_data = None 
        self.m_meta_path = ""
        self.m_file_name = BaseFaceMeta.DEFAULT_META_NAME


    @property
    def file_path(self):
        return osp.join(self.m_meta_path, self.m_file_name)


    def load_from_file(self, pth):
        if osp.isdir(pth):
            self.m_meta_path = pth 
            self.m_file_name = BaseFaceMeta.DEFAULT_META_NAME
        elif osp.isfile(pth):
            self.m_file_name = osp.basename(pth)
            self.m_meta_path = osp.dirname(pth)


        meta_path = self.file_path

        with open(meta_path, 'r') as fp:
            self.m_raw_data = yaml.load(fp, yaml.FullLoader)
    

    # key sep is .
    def __getitem__(self, key):
        pass 

    def component_name_list(self):
        pass 


class IctFaceMeta(BaseFaceMeta):
    def __init__(self):
        super().__init__()
        self.m_components_hierachy = {} 
        self.m_full_index = None 

    def load_from_file(self, pth):
        super().load_from_file(pth)
        meta = self.m_raw_data['meta']['ict_landmark_index']
        keys = list(meta.keys())
        keys = list(filter(lambda x : x != "full_index", keys))


        self.m_full_index = meta['full_index']
        for key in keys:
            item = BaseFaceMeta.ComponentMetaItem()
            item.deserialize(key, meta[key])
            self.m_components_hierachy[key] = item
            






    def __getitem__(self, key):
        meta = self.m_raw_data['meta']['ict_landmark_index']
        return meta[key]


    def component_name_list(self):
        meta = self.m_raw_data['meta']['ict_landmark_index']
        keys = list(meta.keys())
        keys = list(filter(lambda x : x != "full_index", keys))
        return keys 
        



if __name__ == "__main__":
    face_meta = IctFaceMeta()
    face_meta.load_from_file("ict_lmk_info.yaml")
    print("test")