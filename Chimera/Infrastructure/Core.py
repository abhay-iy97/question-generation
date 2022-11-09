from Chimera.Infrastructure.Logging.Logger import *
from Chimera.Infrastructure.ConfigInjector import *

class CoreClasses:
    def __init__(self):
        self.logger = Logger
        self.ConfigType = StaticConfigInjector


def FromConfig(config):
    optLogging = {'native': NativeLogger, 'print': PrintLogger}
    logger = optLogging.get(config['logger'], NativeLogger)()
    optInjection = {'explict' : ExplicitConfigInjector, 'implicit' : ImplicitConfigInjector, 'static' : StaticConfigInjector}
    Core.logger = logger
    Core.ConfigType = optInjection.get(config['config_type'], StaticConfigInjector)
    ConfigInjectorBase.config = config
    Core.ConfigType.Initialize()


Core = CoreClasses()
