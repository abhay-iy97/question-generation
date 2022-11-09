
from Infrastructure.Config.ArgResolver import ArgResolver
argResolver = ArgResolver()
config = argResolver.Parse()

from DataLoad.ChimeraDataset import ChimeraDataset
from DataLoad.DataLoader import DataLoader
from Models.ModelManager import ModelManager
from Models.Engine import Engine


dataset = ChimeraDataset.FromConfig(config)
dataLoader = DataLoader(dataset)
modelManager = ModelManager()
engine = Engine()
engine.Run(modelManager.model, dataLoader)
