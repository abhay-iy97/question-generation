from torch.utils.data import DataLoader as torchDataLoader
from Chimera.Infrastructure.Core import Core
from Chimera.DataLoad.ChimeraDataset import ChimeraDataset


class DataLoader(Core.ConfigType):
    def __init__(self, dataset):
        super().__init__()
        self.loader = torchDataLoader(dataset, batch_size=self.batch_size)  # todo define config
        self.loaderIter = iter(self.loader)

    def GetBatch(self):
        return next(self.loaderIter)  # Idk what's the best way we go about it
