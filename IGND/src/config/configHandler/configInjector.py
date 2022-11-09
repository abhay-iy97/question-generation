
class ConfigInjector(object):
    config = {}

    def __init__(self, configList):
        for var in configList:
            if var in configList:
                setattr(self, var, ConfigInjector.config[var])
            else:
                print(f'warning: {self.__class__.__name__} requested config {var}, but it is not found in config')
