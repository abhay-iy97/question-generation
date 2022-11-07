from Chimera.Infrastructure.Core import Core


class Engine(Core.ConfigType):
    def __init__(self):
        super().__init__()

    def Run(self, model, dataset):
        for epoch in range(self.max_epochs):
            for batch in dataset.loader:
                model(batch)

