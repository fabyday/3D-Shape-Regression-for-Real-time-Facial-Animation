import enum 


class NotCompatibleExtension(Exception):
    pass 

class ImageExtension(enum.Enum):
    pass 


class Category:
    g_category_list = []

    class NotCompatibleCategory(Exception):
        pass 

    def __init__(self, category : str):
        if category in Category.g_category_list    :
            self.m_category = category 
        else:
            self.m_category = ""

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Category):
            pass 
        else :
            raise Exception("object is Not Category")

        if self.m_category == value.m_category:
            return True
        return False


class BaseMeta:
    def __init__(self, type_name = "BaseMeta"):
        self.m_type = type_name
    
    def open_meta():
        pass

    def write_meta():
        pass 

class LandmarkMeta(BaseMeta):
    def __init__(self):
        pass 

    def open_meta(self):
        pass 

    def write_meta(self):
        pass

class ImageMeta(BaseMeta):
    def __init__(self):
        pass 

    def open_meta(self):
        pass 

    def write_meta(self):
        pass