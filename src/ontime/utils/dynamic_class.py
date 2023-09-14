from ..utils import Registry


class DynamicClass(Registry):
    def __init__(self):
        super().__init__()

    def load(self, name, my_class):
        """
        Load a class dynamically
        :param name: str
        :param my_class: the class to load
        """
        self.register(name, my_class)
        self.__setattr__(name, my_class)

    def load_registry(self):
        """
        Load all classes stored in the registry
        """
        for name, my_class in self.registry.items():
            self.__setattr__(name, my_class)
