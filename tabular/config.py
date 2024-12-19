import pandas as pd
import torch


class Config:
    def __init__(self):
        self.train_path = r"/mnt/windows/Users/lordh/Documents/Svalbard/Data/skip_connections/adult/train.csv"
        self.test_path = r"/mnt/windows/Users/lordh/Documents/Svalbard/Data/skip_connections/adult/test.csv"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 128
        self.lr = 0.001
        self.num_epochs = 100
        self.model_name = "NN_1"
        self.do_rotation = False
        self.do_permutation = False
        self.permutation = torch.randperm(107)  # since there are 107 attributes
        self.skip_connection_rotation = 52
        self.description = (
            f"rot-skip-{self.skip_connection_rotation}"
            if self.do_rotation
            else ("perm" if self.do_permutation else "wt-dec-0")
        )
        self.save_model = True
        self.results_dir = r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/skip_connections/runs/"
        self.results_file = r"{}_{}_bsz-{}_lr-{}_ep-{}-desc-{}{}"
        self.capture_weights = True  # this is used to store weights per epoch
        self.use_weight_decay = False
        self.weight_decay = 5 * pow(10, -4)

        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_columns", None)
