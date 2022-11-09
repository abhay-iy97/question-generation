from torch.nn import GRU, Module
from Chimera.Infrastructure.Core import Core
from Chimera.Models.ChimeraModels.Basic.BasicModelAdapter import BasicModelAdapter


class BasicChimera(Module, Core.ConfigType):
    def __init__(self):
        Module.__init__(self)
        self.network = GRU(input_size=self.input_embedding_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers)

    def forward(self, x, h):
        return self.network(x, h)

    @staticmethod
    def WithAdapter():
        return BasicModelAdapter(BasicChimera())
