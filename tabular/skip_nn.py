import torch
import torch.nn as nn

class SkipNN_1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc_1 = nn.Linear(in_features=107, out_features=107)
        self.act_1 = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=107, out_features=2)

    def forward(self, x):
        x1 = self.fc_1(x)
        x1 = self.act_1(x)
        # x2 = torch.roll(x, shifts=self.config.skip_connection_rotation, dims=1)
        # x2 = x[:, config.permutation]
        x = self.fc_2(x1 + x)
        return x

