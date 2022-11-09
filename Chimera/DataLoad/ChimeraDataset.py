
import pandas as pd
import torch
from torch.utils.data import Dataset
from Chimera.Infrastructure.Core import Core
from Chimera.DataLoad.Preprocessor import Preprocessor

class ChimeraDataset(Dataset):
    def __init__(self, df):
        columns = ['context', 'question', 'answer', 'generated_qn'] #todo col names from config?
        if not all([c in df.columns for c in columns]):
            Core.logger.Error('dataset does not comply with the needed format')
        prepro = Preprocessor()
        df = prepro.Prepro(df)
        self.df: pd.DataFrame = df

    @staticmethod
    def FromCSV(path):
        data = pd.read_csv(path)
        return ChimeraDataset(data)

    @staticmethod
    def FromConfig(config):
        loadType = config['load_type']
        path = config['data_path']
        if loadType == 'csv': #todo - better
            return ChimeraDataset.FromCSV(path)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if idx is torch.Tensor:
            idx = idx.tolist()
        return self.df[idx]
