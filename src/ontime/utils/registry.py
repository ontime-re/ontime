class Registry:
    def __init__(self):
        self.registry = {}

    def register(self, name, item):
        self.registry[name] = item

    def get(self, name):
        return self.registry[name]

    def get_all(self):
        return list(self.registry.keys())

    def update(self, name, item):
        self.registry[name] = item

    def delete(self, name):
        del self.registry[name]
