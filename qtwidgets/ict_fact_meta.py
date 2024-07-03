import os  

import os.path as osp 
import yaml 
import typing 
import enum 

import igl
class ComponentEnum(enum.Enum):
    INNER_UPPER=0
    INNER_LOWER=1
    OUTER_UPPER=2
    OUTER_LOWER=3
    UPPER = 2
    LOWER = 3
    VERTICAL = 3
    HORIZONTAL = 3
    JUST_LINE = 3
    NOT_A_END_COMPOENT = 8

class BaseFaceMeta:



    class ComponentMetaItem():

        COMPONENT_ENUM = {"inner":0}
        def __init__(self, name = "", root_flag = True):
            self.m_name = name
            self.m_data = {}
            self.m_indice_list = []
            self.m_root_flag =  root_flag
        @property
        def name(self):
            return self.m_name

        def get_type(self):
            if self.m_name.find("inner_upper") >= 0:
                return ComponentEnum["INNER_UPPER"]
            elif self.m_name.find("inner_lower") >= 0:
                return ComponentEnum["INNER_LOWER"]
            elif self.m_name.find("outer_upper") >= 0:
                return ComponentEnum["OUTER_UPPER"]
            elif self.m_name.find("outer_lower") >= 0:
                return ComponentEnum["LOWER"]
            elif self.m_name.find("lower") >= 0:
                return ComponentEnum["LOWER"]
            elif self.m_name.find("upper") >= 0:
                return ComponentEnum["UPPER"]
            elif self.m_name.find("vertical") >= 0:
                return ComponentEnum["VERTICAL"]
            elif self.m_name.find("horizontal") >= 0:
                return ComponentEnum["HORIZONTAL"]
            else : 
                if self.is_end_component():
                    return ComponentEnum["JUST_LINE"]
                else :
                    return ComponentEnum["NOT_A_END_COMPOENT"]

        
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
                item = BaseFaceMeta.ComponentMetaItem(root_flag=False)
                item.deserialize(key, meta[key])
                self.m_data[key] = item

        def get_component_name(self):
            return list(self.m_data.keys())
        
        def is_end_component(self):
            return True if len(self.m_data.items()) == 0 else False
      
        def keys(self):
            return self.m_data.keys() 
        
        def get_indice_list(self): 
            """
                append child list
            """

            if len(self.m_indice_list):
                return self.m_indice_list
            
            if len(self.m_data) : 
                indice = [] 
                for key, item in self.m_data:
                    indice += item.get_indice_list()
                return list(set(indice))
            return []
        

        def __getitem__(self, key : str): 
            return self.m_data[key]

        def __iter__(self):# shallow iter
            pass 
        
        def get_vertex_contained_index_hierachy(self, v_index):
            pass 
        
        def __contains__(self, item):
            if len(self.m_indice_list) != 0 :
                if item in self.m_indice_list:
                    return True 
                else : 
                    pass 
            else :
                if not self.is_end_component():
                    pass
        def __is_children_linked(self):
            pass 

        def find_linked_index(self, v1_index, v2_index):
            existence_check = True
            if self.m_root_flag:
                check1 = v1_index in self.m_indice_list
                check2 = v2_index in self.m_indice_list
                existence_check = check1 and check2 

            
            v_contained_key = ("", "")
            if existence_check:
                #check cateogry where v idnex is laid.
                for key, item in self.m_data:
                    if v1_index in item:
                        v_contained_key[0] = key
                    if v2_index in item : 
                        v_contained_key[1] = key
            
            else :
                return []

            if v_contained_key[0] == v_contained_key[1]:
                self.m_data[v_contained_key[0]].find_linked_index(v1_index, v2_index)
            
            # self
            


            check = v1_index in self.m_indice_list
            cateogry_v1_contained = ""
            category_v2_contained = "" 



            
    
    DEFAULT_META_NAME = "meta.yaml"
    def __init__(self):
        self.m_raw_data = None 
        self.m_meta_path = ""
        self.m_file_name = BaseFaceMeta.DEFAULT_META_NAME

        self.m_template_mesh = None  # (v,f) tuple
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
    
    def __len__(self):
        return NotImplemented
    # key sep is .
    def __getitem__(self, key):
        return NotImplemented

    def component_name_list(self):
        return NotImplemented

    def get_full_index(self):
        return NotImplemented
    

    def get_template_mesh(self):
        return self.m_template_mesh[0]



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

        self.m_landmark_vertex_index = self.m_raw_data['meta']['ict_landmark_vertex_indice']
        generic_mesh_file = os.path.join(self.m_meta_path, self.m_raw_data['meta']['ict_generic_face_file'])
        v, f = igl.read_triangle_mesh(generic_mesh_file)
        v = v[self.m_landmark_vertex_index, :]
        self.m_template_mesh = (v,f)
        
    
    def get_full_index(self):
        return self.m_full_index

    def __len__(self):
        return len(self.m_landmark_vertex_index)

    def __getitem__(self, key):
        return self.m_components_hierachy[key]


    def component_name_list(self):
        meta = self.m_raw_data['meta']['ict_landmark_index']
        keys = list(meta.keys())
        keys = list(filter(lambda x : x != "full_index", keys))
        return keys 
    


if __name__ == "__main__":
    face_meta = IctFaceMeta()
    face_meta.load_from_file("ict_lmk_info.yaml")
    print("test")