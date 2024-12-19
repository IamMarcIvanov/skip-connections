import torch

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    nn_path_1: str = r"./runs/0017_NN_1_bsz-32_lr-0.001_ep-200-desc-.pth"
    nn_path_2: str = r"./runs/0016_NN_1_bsz-1024_lr-0.001_ep-200-desc-.pth"
    device = torch.device("cpu")
    num_points = 2000
