from Chimera.Infrastructure.Core import Core
from torch.nn import Module

class ModelAdapterBase(Core.ConfigType):
    def __init__(self, model):
        self.model : Module = model

    def RunBatch(self, batch):
        pass

