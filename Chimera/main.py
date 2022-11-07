
from Infrastructure.Config.Config import FromArgs
from Infrastructure.Core import FromConfig
config = FromArgs()
FromConfig(config)

from DataLoad.ChimeraDataset import ChimeraDataset
from DataLoad.DataLoader import DataLoader


dataset = ChimeraDataset.FromConfig(config)
dataLoader = DataLoader(dataset)
