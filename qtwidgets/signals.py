import enum 
class EventType(enum.Enum):
    DATA_UPDATED = 1
    ALL_DATA_SAVED = 2
    ALL_LANDMARK_DETECTED = 3
    SELECTED_DATA_CHNAGED = 4
    DATA_LOADED_FROM_META = 5
    

class Event():
    def __init__(self, *args):
        self.events = [ arg.value for arg in args]

    def __contains__(self, item : EventType):
        if item.value in self.events:
            return True 
        return False 
    


if __name__ == "__main__":
    e = Event(EventType.DATA_LOADED_FROM_META, EventType.ALL_LANDMARK_DETECTED)
    if EventType.DATA_LOADED_FROM_META in e :
        print("t")

    if EventType.ALL_LANDMARK_DETECTED in e : 
        print("test")

    if EventType.DATA_UPDATED not in e : 
        print("test")
        






