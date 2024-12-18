import pandas as pd
from rich.pretty import pprint
from rich.table import Table
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchviz import make_dot

import inspect
import math
import os

from .salary_dataset import SalaryDataset


class Util:
    def __init__(self, config):
        self.config = config
        self.train_set, self.test_set, self.train_loader, self.test_loader = (
            None,
            None,
            None,
            None,
        )
        self.num_classes = 2
        self.file_num = self.get_file_num()

        self.set_data()

    def set_data(self):
        self.train_set = SalaryDataset(self.config, train=True)
        self.test_set = SalaryDataset(self.config, train=False)
        self.train_loader = DataLoader(
            self.train_set, batch_size=self.config.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_set, batch_size=self.config.batch_size, shuffle=False
        )

    def get_file_num(self):
        files = os.listdir(self.config.results_dir)
        if not files:
            run_num = "0"
        else:
            run_num = str(max([int(file[:4]) for file in files]) + 1)
        num_zeros_prefix = 4 - len(run_num)
        return "0" * num_zeros_prefix + run_num

    def get_file_path(self, file_type=".txt"):
        results_file = self.config.results_file.format(
            self.file_num,
            self.config.model_name,
            self.config.batch_size,
            self.config.lr,
            self.config.num_epochs,
            self.config.description,
            file_type,
        )
        return self.config.results_dir + results_file

    def get_l_out(self, l_in, padding=0, dilation=1, kernel_size=None, stride=1):
        num = l_in + 2 * padding - dilation * (kernel_size - 1) - 1
        res = math.floor((num / stride) + 1)
        return res

    def init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def write_forward_method(self, model):
        print(inspect.getsource(model.forward))

    def plot_architecture(self, path, model):
        random_input = torch.rand(1, 107).to(self.config.device)
        output = model(random_input)
        dot = make_dot(output, params=dict(model.named_parameters()))
        dot.render(filename=path)

    def write_param_count(self, model, console):
        total_params = 0
        table = Table("name", "count", title="param count")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            num_params = param.numel()
            table.add_row(name, str(num_params))
            total_params += num_params
        console.print(table)
        print("Total Number of Parameters:", total_params)

    def analyse_data(self):
        # print(len(util.train_set))
        train = True
        train_X = pd.read_csv(config.train_path, header=None)
        test_X = pd.read_csv(config.test_path, header=None)
        train_Y = train_X[14]
        test_Y = test_X[14]
        train_X.drop(14, axis="columns", inplace=True)
        test_X.drop(14, axis="columns", inplace=True)
        train_one_hot = pd.get_dummies(train_X, dtype=float)
        test_one_hot = pd.get_dummies(test_X, dtype=float)
        train_X = (train_one_hot - train_one_hot.mean()) / train_one_hot.std()
        test_X = (test_one_hot - test_one_hot.mean()) / test_one_hot.std()
        data = train_X if train else test_X
        labels = train_Y if train else test_Y
        pprint(torch.tensor(data.iloc[302].values, dtype=torch.float32))
        pprint(torch.tensor(labels.iloc[302], dtype=torch.long))
