from torch.utils.data import Dataset
import pandas as pd
from torch import tensor
import torch


class ClassificationDataset(Dataset):
    def __init__(self, data_file):
        super().__init__()
        self.df = pd.read_csv(data_file)
        self.df_len = self.df.shape[0]
        self.features_columns = sorted([col for col in self.df.columns if "feature" in col])
        self.id_column = "id"
        self.target_column = "target"
        
    def __len__(self):
        return self.df_len
    
    def __getitem__(self, idx):
        
        features = self.df[self.features_columns].values[idx]
        object_id = self.df[self.id_column].values[idx]
        target = self.df[self.target_column].values[idx]
        
        return tensor(object_id, dtype=torch.int32), {
            "features" : tensor(features, dtype=torch.float32)
        }, tensor(target, dtype=torch.int64)
        

class RegressionDataset(Dataset):
    def __init__(self, data_file):
        super().__init__()
        self.df = pd.read_csv(data_file)
        self.df_len = self.df.shape[0]
        self.features_columns = sorted([col for col in self.df.columns if "feature" in col])
        self.id_column = "id"
        self.target_column = "target"
        
    def __len__(self):
        return self.df_len
    
    def __getitem__(self, idx):
        
        features = self.df[self.features_columns].values[idx]
        object_id = self.df[self.id_column].values[idx]
        target = self.df[self.target_column].values[idx]
        
        return tensor(object_id, dtype=torch.int32), {
            "features" : tensor(features, dtype=torch.float32)
        }, tensor([target], dtype=torch.float32)
        