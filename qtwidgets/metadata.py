import enum 

import os 
import os.path as osp 
import yaml 
import inspect
import uuid
from typing import Union, Optional
import logger



meta_logger = logger.root_logger.getChild("metadata")



def is_meta_file(path):
    if osp.isfile(path) and osp.splitext(osp.basename(path))[-1] == "yaml": # check extension file
        return True 
    return False

class Category:
    pass 
class BaseItemMeta():
    class ItemMetaKeyError(Exception):
        pass
    def __init__(self):
        self.m_uuid =  uuid.uuid4()

    @property
    def unique_id(self):
        return self.m_uuid
    
    def serialize(self):
        raise NotImplementedError("Not implemented serialize method")

    def deserialize(self):
        raise NotImplementedError("Not implemented deserialize method")


    @property
    def category(self):
        return self.m_parent_category
    
    @category.setter
    def category(self, category : Category):
        self.m_parent_category = category

class NotCompatibleExtension(Exception):
    pass 

class ImageExtensionEnum(enum.Enum):
    JPG =  ".JPG"
    JPEG = ".JPEG"
    PNG =  ".PNG"


#forward decl
class Category:
    pass
class CategoryCollection():
    pass

class CategoryCollection():

    class CategoryCollectionIterator():
        def __init__(self, collection_obj : CategoryCollection):
            self.m_length = len(collection_obj.keys())
            self.cur_idx = 0
            self.m_ref_collection_obj = collection_obj
            self.m_cur_sub_iterator = iter(self.m_ref_collection_obj[self.cur_idx])
        def __iter__(self):
            return self
        
        def __next__(self):
            try:
                return next(self.m_cur_sub_iterator)
            except StopIteration:
                self.cur_idx += 1
                if self.cur_idx >= self.m_length:
                    raise StopIteration()
                self.m_cur_sub_iterator = iter(self.m_ref_collection_obj[self.cur_idx])
                return next(self.m_cur_sub_iterator)
            
    class CategoryCollectionIterator2(): # shallow iterator 
        def __init__(self, collection_obj : CategoryCollection):
            self.m_length = len(collection_obj.keys())
            self.cur_idx = 0
            self.m_ref_collection_obj = collection_obj
            self.m_cur_sub_iterator = iter(self.m_ref_collection_obj[self.cur_idx])

        def __iter__(self):
            return self 
        
        def __next__(self):
            if self.cur_idx >= self.m_length:
                raise StopIteration()
            res = self.m_ref_collection_obj[self.cur_idx]
            self.cur_idx += 1
            return res
            
    class CategoryAlreadyExistsException(Exception):
        pass

    def __init__(self, item_cls : type):
        self.m_categories = {}
        self.m_item_cls = item_cls



    def append_category(self, category_obj: Category):
        category_item = self.m_categories.get(category_obj.m_category_name, None)
        if category_item is not None :
            raise CategoryCollection.CategoryAlreadyExistsException("same name of category is already exists in collection")
        self.m_categories[category_obj.m_category_name] = category_obj


    def serialize(self):
        res = {}
        for _, item in self.m_categories.items():
            res[item.m_category_name] = item.serialize()
        return res


    def deserialize(self, data : dict):
        for key, item in data.items():
            cat = Category(key, self, self.m_item_cls)
            cat.deserialize(item)            

    def keys(self):
        return self.m_categories.keys()
    
    def __getitem__(self, key_o_idx):
        if isinstance( key_o_idx, str):
            return self.m_categories[key_o_idx]
        elif isinstance(key_o_idx, int):
            key = list(self.keys())[key_o_idx]
            return self.m_categories[key]


    def __iter__(self):
        return CategoryCollection.CategoryCollectionIterator2(self)


class Category:
    g_category_list = []

    class CategoryIterator():
        def __init__(self,collection_obj:Category):
            self.m_length = len(collection_obj.keys())
            self.cur_idx = 0
            self.m_ref_collection_obj = collection_obj

        def __iter__(self):
            return self 
        
        def __next__(self):
            if self.cur_idx < self.m_length:
                res = self.m_ref_collection_obj[self.cur_idx]
            else :
                raise StopIteration()
            
            self.cur_idx += 1
            return res
    class NotCompatibleCategory(Exception):
        pass 
   

    def __init__(self, category_name : str, category_collection : CategoryCollection, cls : type):
        self.m_category_name = category_name
        self.m_cls = cls
        self.m_items = {}
        self.m_global_category_collection  = category_collection
        self.m_global_category_collection.append_category(self)

    @property
    def category_name(self):
        return self.m_category_name

    def add_item(self, item : BaseItemMeta):
        if isinstance(item, self.m_cls): 
            self.m_items[item.unique_id] = item
            item.m_parent_category = self
        else:
            raise 
    def __iter__(self):
        return Category.CategoryIterator(self)

    def __eq__(self, value: Category) -> bool:
        
        if isinstance(value, Category):
            if value is self :
                return True 
        else :
            raise Exception("object is Not Category")

        if self.m_category_name == value.m_category_name:
            return True
        return False
    
    def keys(self):
        return self.m_items.keys()

    def __getitem__(self, key_o_idx):
        if isinstance( key_o_idx, uuid.UUID):
            return self.m_items[key_o_idx]
        elif isinstance(key_o_idx, int):
            key = list(self.keys())[key_o_idx]
            return self.m_items[key]

    
    def serialize(self):
        result = []
        for _, item in self.m_items.items():
            result.append(item.serialize())
        return result

    def deserialize(self, raw_data_list : list):
        for data in raw_data_list:
            obj = self.m_cls()
            obj.deserialize(data, self)
            self.m_items[obj.unique_id]=obj
        


class BaseMeta:
    Default_File_Name = "meta.yaml"
    CATEGORY_DATA_KEY = "meta"

    class CategoryDataKeyNotExistsException(Exception):
        pass
    class ItemMetaIsNotUnique(Exception):
        pass
    class RawDataNotLoadedException(Exception):
        pass 
    
    class MetaTypeNotCompatibleException(Exception):
        pass
    def __init__(self, type_name = "BaseMeta", cls_type : type = None):
        self.m_type = type_name
        self.m_cls_type = self.get_item_meta() if cls_type is None else cls_type
        self.reset()



    def __getitem__(self, item_uuid):
        for category in self.m_category_collection:
            try: 
                item = category[item_uuid]
            except:
                continue 
        return item




    def get_item_meta(self):
        i = 0
        for info in dir(self):
            try :
                obj = getattr(self, info)
            except : 
                continue # if exteipns is occur, it is not object.
            if isinstance(obj, type) and issubclass(obj, BaseItemMeta):
                i += 1
                meta = obj
        if not i == 1:
            raise BaseMeta.ItemMetaIsNotUnique("meta Item is Not Unique. Can't measure it which to Use.")
        return meta


    def get_item_iterator(self):
        return CategoryCollection.CategoryCollectionIterator(self.m_category_collection)
    
    def get_category_iterator(self):
        return CategoryCollection.CategoryCollectionIterator2(self.m_category_collection)

    def get_category_data_key(self):
        try:
            key = getattr(self, "CATEGORY_DATA_KEY")
        except:
            raise BaseMeta.CategoryDataKeyNotExistsException()
        return key



    def create(self, filename : Optional[str] = None  ):
        self.reset(filename)
        self.m_raw_data = {"meta" : {"meta_type" : self.meta_type}}
        self.m_raw_data_loaded = True
        self.m_category_collection = CategoryCollection(self.m_cls_type)
    

    def reset(self, filename :Optional[str] = None ):
        self.m_meta_file_name = BaseMeta.Default_File_Name if filename is None else filename
        self.m_raw_data = None 
        self.m_location = "" 
        self.m_raw_data_loaded = False
        self.m_category_collection = CategoryCollection(self.m_cls_type)

    @property
    def is_loaded(self):
        return self.m_raw_data_loaded

    @property
    def raw_data(self):
        if self.m_raw_data_loaded:
            return self.m_raw_data
        raise BaseMeta.RawDataNotLoadedException()
    
    @property 
    def meta_type(self):
        return self.m_type

    @property
    def file_location(self):
        return self.m_location
    
    def _meta_type_check(self):
        if not (self.meta_type == self.m_raw_data['meta'].get('meta_type', self.meta_type)): # if not exists meta_type skip 
            raise BaseMeta.MetaTypeNotCompatibleException("Meta type is Not compatible")


    def open_meta(self, path):
        self.m_location = path
        meta_path = path 
        if osp.isdir(meta_path):
            meta_path = osp.join(meta_path, BaseMeta.Default_File_Name)
        elif is_meta_file(meta_path):
            meta_path = meta_path
        else:
            raise BaseMeta.RawDataNotLoadedException("it is not meta file and directory where not contain meta file")
        with open(meta_path, "r") as fp:
            self.m_raw_data = yaml.load(fp, yaml.FullLoader)
        
        self._meta_type_check() # raise exceiption MetaTypeNotCompatibleException
        meta_file_name = osp.basename(meta_path)
        meta_dir_path = osp.dirname(meta_path)

        self.m_location = meta_dir_path 
        self.m_meta_file_name = meta_file_name
        self.m_raw_data_loaded = True
        self.m_category_collection.deserialize(self.data(self.get_category_data_key()))

        return self.m_raw_data_loaded

    def write_meta(self, path = None):
        if path is None :
            save_path = self.m_location
        elif os.path.isdir(path) :
            if not osp.exists(path):
                os.makedirs(path)
            save_path = osp.join(path, BaseMeta.Default_File_Name)
        elif is_meta_file(path):
            if not osp.exists(osp.dirname(path)):
                os.makedirs(osp.dirname(path))
            save_path = path

        self.m_location = save_path
        dict_items = self.m_category_collection.serialize()
        self.raw_data['meta']["images_name"] = dict_items
        with open(save_path, "w") as fp: 
            yaml.dump(self.m_raw_data, fp, sort_keys=False)
        return True 
    

    def convert_to(self):
        raise NotImplementedError("Not implement convert_to method.")


    def add_category(self, category_name : str):
        try : 
            Category(category_name, self.m_category_collection, self.m_cls_type)
        except:
            pass

        

    def add_item_to_category(self, item : BaseItemMeta, cateogry_name : Union[str , Category]):
        pass 

    def data(self, key:str):
        """
            sep : .
            ex ) meta.images_name
        """
        splited_keys = key.split(".")
        cur_data = self.m_raw_data
        for key in splited_keys:
            cur_data = cur_data[key]
        return cur_data


class LandmarkMeta(BaseMeta):
    META_NAME = "Landmark"
    CATEGORY_DATA_KEY = "meta.images_name"

    class LandmarkItemMeta(BaseItemMeta):
        LANDMAKR_KEY = "landmark"
        NAME_KEY = "name"
        def __init__(self):
            super().__init__()
            self.m_parent_category = None
            self.m_landmark_name = ""
            self.m_name = ""
        
        @property
        def landmark(self):
            return self.m_landmark_name
        
        @landmark.setter
        def landmark(self, name):
            self.m_landmark_name = name 
        



        @property 
        def name(self):
            return self.m_name

        @name.setter
        def name(self, name):
            self.m_name = name

        def serialize(self):
            return {LandmarkMeta.LandmarkItemMeta.LANDMAKR_KEY : self.m_landmark_name, 
                     LandmarkMeta.LandmarkItemMeta.NAME_KEY: self.m_name}

        def deserialize(self, data : dict, parent_category : Category = None):
            self.m_parent_category = parent_category
            lmk_name = data.get(LandmarkMeta.LandmarkItemMeta.LANDMAKR_KEY, None)
            if lmk_name is None :
                raise BaseItemMeta.ItemMetaKeyError('"landmark" key is not existed.')
            self.m_landmark_name = lmk_name

            name = data.get(LandmarkMeta.LandmarkItemMeta.NAME_KEY, None)

            if name is None :
                raise BaseItemMeta.ItemMetaKeyError('"name" key is not existed.')
            self.m_name = data.get(LandmarkMeta.LandmarkItemMeta.NAME_KEY, None)

    def __init__(self):
        super().__init__(LandmarkMeta.META_NAME)
    
    def create(self):
        super().create()
        # self.m_raw_data['meta']['images_name'] = dict()

    def open_meta(self, path):
        super().open_meta(path)

    def _meta_type_check(self):
        super()._meta_type_check()
        if not self.m_raw_data['meta'].get('images_root', False): # if images_root not exists in meta file . it is image meta.
            raise BaseMeta.MetaTypeNotCompatibleException("this is not landmark meta")

    def write_meta(self, path = None):
        """
            if path is None overwrite meta
        """
        super().write_meta(path)

    def keys(self):
        if self.is_loaded:
            return self.raw_data['meta']['images_name'].keys() 
        raise BaseMeta.RawDataNotLoadedException("not loaded exception")
    @property
    def extension(self):
        if self.is_loaded:
            ext_str = self.raw_data['meta']['file_ext']
            ext_str = ext_str if ext_str.startswith(".") else "."+ext_str
            return ImageExtensionEnum(ext_str)
        raise BaseMeta.RawDataNotLoadedException("not loaded exception")
    @extension.setter
    def extension(self, ext : Union[str , ImageExtensionEnum]):
        if isinstance(ext, ImageExtensionEnum):
            self.raw_data['meta']['file_ext'] = ext.value
        elif isinstance(ext, str) : 
            if ext.startswith("."):
                self.raw_data['meta']['file_ext'] = ext
            else : 
                self.raw_data['meta']['file_ext'] = "."+ext


    @property
    def image_location(self):
        return self.raw_data['meta']['images_root']
    
    @image_location.setter
    def image_location(self, root_pth):
        self.raw_data['meta']['images_root'] = root_pth
    
    def __iter__(self):
        return iter(self.m_category_collection)

    def convert_to(self):
        cvt_image_meta = ImageMeta()
        cvt_image_meta.reset()
        for category_item in self:
            category = Category(category_item.category_name, cvt_image_meta.m_category_collection, cvt_image_meta.m_cls_type)
            for item in category_item:
                res = ImageMeta.ImageItemMeta()
                res.name = item.name
                category.add_item(res)


class ImageMeta(BaseMeta):
    META_NAME = "Image"
    CATEGORY_DATA_KEY = "meta.images_name"
    class ImageItemMeta(BaseItemMeta):
        def __init__(self):
            super().__init__()
            self.m_parent_category = None
            self.m_name = ""
        
        @property
        def name(self):
            return self.m_name
        
        @name.setter
        def name(self, name):
            self.m_name = name
        
        def serialize(self):
            return self.name 
        
        def deserialize(self, data, parent_category  :Category = None):
            self.name = data
            self.m_parent_category = parent_category


    def __init__(self):
        super().__init__(ImageMeta.META_NAME)
        self.create()

    def create(self):
        super().create()
        self.m_raw_data['meta']['file_ext'] = ""
        # self.m_raw_data['meta']['images_name'] = dict()
        

    def open_meta(self, path):
        super().open_meta(path)

    def _meta_type_check(self):
        super()._meta_type_check()
        if not self.m_raw_data['meta'].get('images_root', True): # if images_root not exists in meta file . it is image meta.
            raise BaseMeta.MetaTypeNotCompatibleException("this is not images_meta meta")

    def write_meta(self, path = None):
        super().write_meta(path)
    @property
    def extension(self):
        if self.is_loaded:
            ext_str = self.raw_data['meta']['file_ext']
            ext_str = ext_str if ext_str.startswith(".") else "."+ext_str
            return ImageExtensionEnum(ext_str)
        raise BaseMeta.RawDataNotLoadedException("not loaded exception")
    
    @extension.setter
    def extension(self, ext : Union[str , ImageExtensionEnum]):
        if isinstance(ext, ImageExtensionEnum):
            self.raw_data['meta']['file_ext'] = ext.value
        elif isinstance(ext, str) : 
            if ext.startswith("."):
                self.raw_data['meta']['file_ext'] = ext
            else : 
                self.raw_data['meta']['file_ext'] = "."+ext

    def __iter__(self):
        return iter(self.m_category_collection)

    def keys(self):
        if self.is_loaded:
            return self.raw_data['meta']['images_name'].keys() 
        raise BaseMeta.RawDataNotLoadedException("not loaded exception")

    def convert_to(self):
        cvt_lmk_meta = LandmarkMeta()
        cvt_lmk_meta.create()
        cvt_lmk_meta.extension = self.extension
        cvt_lmk_meta.image_location = self.file_location
        for category_item in self:
            category = Category(category_item.category_name, cvt_lmk_meta.m_category_collection, cvt_lmk_meta.m_cls_type)
            for item in category_item:
                res = LandmarkMeta.LandmarkItemMeta()
                res.name = item.name
                res.landmark = item.name + "_lmk.txt"
                category.add_item(res) 
        return cvt_lmk_meta
if __name__ == "__main__":
    image_meta = ImageMeta()
    image_meta.open_meta(osp.join(osp.dirname(osp.dirname(__file__)), "images/all_in_one/expression"))
    lmk_meta = LandmarkMeta()
    try:
        lmk_meta.open_meta(osp.join(osp.dirname(osp.dirname(__file__)), "images/all_in_one/expression"))
    except:
        print("err")
    print(ImageExtensionEnum.JPEG.name)
    print(ImageExtensionEnum.JPEG.value)
    print(ImageExtensionEnum.JPEG == ImageExtensionEnum("JPEG"))
    print(ImageExtensionEnum.JPG.value == "JPG")
    print(ImageExtensionEnum.JPEG.value == 2)