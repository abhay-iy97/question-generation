class ConfigInjectorBase(object):
    config = {}

    def __init__(self):
        pass

    @staticmethod
    def Initialize():
        pass


class ExplicitConfigInjector(ConfigInjectorBase):
    def __init__(self, configList):
        for var in configList:
            if var in configList:
                setattr(self, var, ConfigInjectorBase.config[var])
            else:
                print(f'warning: {self.__class__.__name__} requested config {var}, but it is not found in config')


class ImplicitConfigInjector(ConfigInjectorBase):
    config = {}  # config is defined as 'config_key' : {"value" : val, "class": [className])

    def __init__(self):
        for c, val in ConfigInjectorBase.config.items():

            if self.__class__.__name__ not in val['class']:
                continue
            setattr(self, val['class'], val['value'])


class StaticConfigInjector(ConfigInjectorBase):

    def __init__(self):
        for k, v in ConfigInjectorBase.config.items():
            StaticConfigInjector.k = v

    @staticmethod
    def Initialize():
        for k, v in ConfigInjectorBase.config.items():
            setattr(StaticConfigInjector, k, v)
