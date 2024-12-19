import matplotlib.pyplot as plt
from rich.pretty import pprint
import torch
import torch.nn as nn
from tqdm import tqdm

from loss_visualization.config import Config
from tabular.skip_nn import SkipNN_1
from tabular.basic_nn import NN_1
from tabular.config import Config as TabularConfig
from tabular.util import Util
from tabular.salary_dataset import SalaryDataset


def set_model_weights(model, theta_init, theta_final, alpha):
    for name, param in model.named_parameters():
        value = (1 - alpha) * theta_init[name] + alpha * theta_final[name]
        param.data = nn.parameter.Parameter(value)


def get_weights_from_path(path):
    data = torch.load(path, weights_only=True, map_location=config.device)
    return dict(data)


def get_loss(model, alpha_values, theta_init, theta_final):
    criterion = nn.CrossEntropyLoss()
    train_set = SalaryDataset(tabular_config, train=True)
    test_set = SalaryDataset(tabular_config, train=False)
    train_loss_values, test_loss_values = [], []
    for alpha in tqdm(alpha_values):
        set_model_weights(
            model=model, alpha=alpha, theta_init=theta_init, theta_final=theta_final
        )
        train_output = model(torch.tensor(train_set.data.values, dtype=torch.float32))
        train_loss = criterion(train_output, torch.tensor(train_set.labels, dtype=torch.long))
        train_loss_values.append(train_loss)
        test_output = model(torch.tensor(test_set.data.values, dtype=torch.float32))
        test_loss = criterion(test_output, torch.tensor(test_set.labels, dtype=torch.long))
        test_loss_values.append(test_loss)
    return train_loss_values, test_loss_values


def plot_graph(alpha, loss, label):
    ax.plot(alpha, loss, label=label)


if __name__ == "__main__":
    config = Config()
    tabular_config = TabularConfig()
    util = Util(tabular_config)

    with torch.no_grad():
        alpha_values = torch.linspace(-0.5, 1.5, config.num_points)
        model = NN_1(tabular_config)
        theta_init = get_weights_from_path(path=config.nn_path_1)
        theta_final = get_weights_from_path(path=config.nn_path_2)
        # for name, param in basic_model.named_parameters():
        #     theta_init[name] = nn.parameter.Parameter(3 * torch.rand(param.shape) + 1)
        #     theta_final[name] = nn.parameter.Parameter(3 * torch.rand(param.shape) + 1)
        train_loss, test_loss = get_loss(
            model=model,
            alpha_values=alpha_values,
            theta_init=theta_init,
            theta_final=theta_final,
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_graph(
            alpha_values,
            train_loss,
            label="train",
        )
        plot_graph(
            alpha_values,
            test_loss,
            label="test",
        )
        plt.legend()
        plt.show()
