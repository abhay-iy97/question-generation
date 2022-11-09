from Chimera.Models.ChimeraModels.Base.ModelAdapterBase import ModelAdapterBase


class BasicModelAdapter(ModelAdapterBase):
    def __init__(self, model):
        super().__init__(model)

    def RunBatch(self, batch):
        pass


