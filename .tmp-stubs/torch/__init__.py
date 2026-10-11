class cuda:
    @staticmethod
    def is_available():
        return False

float32 = object()

def set_default_dtype(_):
    return None

class Tensor(list):
    pass

def tensor(data):
    return Tensor(data)
