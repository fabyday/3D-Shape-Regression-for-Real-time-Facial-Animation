import inspect 

import typing
class Parent:

    class BaseMessage():
        MSG_TYPE = "default"
        def __init__(self):
            self.test = 10
            print("test", self.test)
            pass 
        

    def __init__(self):
        return
    

class test(Parent):
    def __init__(self):
        pass 


    class second(Parent.BaseMessage):
        def __init__(self):
            self.test = 102
            print("test", self.test)

class MessageInstanceHelper:
    class MessageBuildFailedException(Exception):
        pass 
    @staticmethod
    def create_message_object_from_nested_class(outer_class, base_class_type, buffer):
        member_list = inspect.getmembers(outer_class)
        sub_class_set = set()
        for (name, member_o) in member_list:
            if inspect.isclass(member_o):
                if  member_o is not base_class_type and issubclass(member_o, base_class_type):
                    sub_class_set.add(member_o)
        for cls in sub_class_set:
            try : 
                msg_obj = cls.create(buffer)
            except Exception as e : 
                msg_obj = None 
                exception = e 
        
        if msg_obj is None :
            raise MessageInstanceHelper.MessageBuildFailedException("message build failed. due to " + exception)
        return msg_obj

    def create_message_object_from_msg_class_candidates(buffer, *cls_types):
        
        for cls in cls_types:
            try:
                msg = cls.create(buffer)
            except Exception as e :
                msg = None 
                exception = e 

        if msg is None :
            raise MessageInstanceHelper.MessageBuildFailedException("Message build fialed. due to " + exception)
        return msg 

if __name__ =="__main__":
    MessageInstanceHelper.create_message_object_from_nested_class(test, Parent.BaseMessage, header=None, data =None)