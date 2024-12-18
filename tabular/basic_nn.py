import torch.nn as nn


class NN_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_1 = nn.Linear(in_features=107, out_features=107)
        self.act_1 = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=107, out_features=2)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.act_1(x)
        x = self.fc_2(x)
        return x
