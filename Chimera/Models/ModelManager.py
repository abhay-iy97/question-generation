from Chimera.Infrastructure.Core import Core
from Chimera.Models.ChimeraModels.Basic.BasicChimera import BasicChimera


class ModelManager(Core.ConfigType):
    def __init__(self):
        models = {'basic': BasicChimera}
        self.model = models.get(self.model_type, BasicChimera).WithAdapter()

    def Write(self):
        pass
