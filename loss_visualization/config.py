from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    skip_nn_path: str = (
        r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/skip_connections/runs/0014_SkipNN_1_bsz-128_lr-0.001_ep-200-desc-.pth"
    )
    basic_nn_path: str = (
        r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/skip_connections/runs/0008_NN_1_bsz-128_lr-0.01_ep-60-desc-.pth"
    )
