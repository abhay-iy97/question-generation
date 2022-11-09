
from Infrastructure.Config.ArgResolver import ArgResolver
argResolver = ArgResolver()
config = argResolver.Parse()

from DataLoad.ChimeraDataset import ChimeraDataset
from DataLoad.DataLoader import DataLoader


dataset = ChimeraDataset.FromConfig(config)
dataLoader = DataLoader(dataset)
