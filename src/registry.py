# registry.py

class ClassRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, cls):
        self._registry[cls.__name__] = cls
        return cls

    def get_class(self, class_name):
        return self._registry.get(class_name)

registry = ClassRegistry()

def register_class(cls):
    return registry.register(cls)
