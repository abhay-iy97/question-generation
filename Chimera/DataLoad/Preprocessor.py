from pandas import DataFrame
import re
from Chimera.Infrastructure.Core import Core


class Preprocessor(Core.ConfigType):
    def __init__(self):
        embeddings = {'naive': self.NaiveEmbedding}
        self.Embedding = embeddings.get(self.embedding_type, self.NaiveEmbedding)

        pass

    def NaiveEmbedding(self, df):
        return df

    def Prepro(self, df):
        df['context'] = self.ApplyContextPrepro(df['context'])
        self.Embedding(df)

    def ApplyContextPrepro(self, context: DataFrame):
        return context.apply(lambda s: re.sub('[^a-zA-Z]+', '', s))
