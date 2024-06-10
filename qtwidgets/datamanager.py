

import os 
import os.path as osp
from qtwidgets import metadata

import yaml
class DataManager:
    def __init__(self):
        self.metadata = None


    def load(self, ins):
        """
            ins : 
                - meta file path(str)
                - meta data(dict)
        """
        if isinstance(ins, str):
            self.load_metadata_file(ins)
        elif isinstance(ins, dict):
            pass
    

    def load_metadata_file(self, path):
        if osp.isdir(path):
            with open(osp.join(path, metadata.metadata_default_filename), "r") as fp:
                loaded_data = yaml.load(fp, yaml.FullLoader)
        elif osp.isfile(path):
            if osp.exists(path):
                with open(path, "r") as fp:
                    loaded_data = yaml.load(fp, yaml.FullLoader)


        self.metadata = loaded_data


class DataCollection:
    def __init__(self):
        self.m_data_items = [] 


class ImageCollection():
    def __init__(self):
        pass






    def save_data_by_name():
        pass 


    def save_data_by_index():
        pass 


    def save_data_by_group():
        pass



class Data:
    def __init__(self):
        self.m_image  = None 
        self.m_landmark = None 
        self.m_category  = None 






class DataLoaderFactory:
    def __init__(self):
        pass 



    @staticmethod
    def load(meta):
        pass 

