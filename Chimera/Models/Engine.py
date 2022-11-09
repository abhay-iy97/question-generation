from Chimera.Infrastructure.Core import Core
from Chimera.Models.ChimeraModels.Base.ModelAdapterBase import ModelAdapterBase


class Engine(Core.ConfigType):
    def __init__(self):
        super().__init__()

    def Run(self, model: ModelAdapterBase, dataloader):
        for epoch in range(self.max_epochs):
            pass



