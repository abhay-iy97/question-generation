from torch.nn import GRU, Module
from Chimera.Infrastructure.Core import Core

class BasicChimera(Module, Core.ConfigType):
    def __init__(self):
        self.network = GRU(input_size=self.input_embedding_size,
                           hidden_size= self.hidden_size,
                           num_layers= self.num_layers) #todo add to config

    def forward(self, x):
        return self.network(x)