from torch.utils.data import Dataset
import pandas as pd
import torch


class SalaryDataset(Dataset):
    def __init__(self, config, train=True):
        self.config = config
        train_X = pd.read_csv(self.config.train_path, header=None)
        test_X = pd.read_csv(self.config.test_path, header=None)
        train_Y = train_X[14]
        test_Y = test_X[14]
        train_X.drop(14, axis="columns", inplace=True)
        test_X.drop(14, axis="columns", inplace=True)
        train_one_hot = pd.get_dummies(train_X, dtype=float)
        test_one_hot = pd.get_dummies(test_X, dtype=float)
        train_X = (train_one_hot - train_one_hot.mean()) / train_one_hot.std()
        test_X = (test_one_hot - test_one_hot.mean()) / test_one_hot.std()
        self.data = train_X if train else test_X
        self.labels = train_Y if train else test_Y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(
            self.data.iloc[idx].values, dtype=torch.float32
        ), torch.tensor(self.labels.iloc[idx], dtype=torch.long)
