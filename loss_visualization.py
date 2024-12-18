import matplotlib.pyplot as plt
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


def get_loss(model, alpha_values, theta_init, theta_final):
    criterion = nn.CrossEntropyLoss()
    train_set = SalaryDataset(tabular_config, train=True)
    loss_values = []
    for alpha in tqdm(alpha_values):
        set_model_weights(
            model=model, alpha=alpha, theta_init=theta_init, theta_final=theta_final
        )
        output = model(torch.tensor(train_set.data.values, dtype=torch.float32))
        loss = criterion(output, torch.tensor(train_set.labels, dtype=torch.long))
        loss_values.append(loss)
    return loss_values


def plot_graph(alpha, loss, label):
    ax.plot(alpha, loss, label=label)


if __name__ == "__main__":
    config = Config()
    tabular_config = TabularConfig()
    util = Util(tabular_config)

    with torch.no_grad():
        alpha_values = torch.linspace(0, 1, 200)
        basic_model = NN_1()
        skip_model = SkipNN_1(tabular_config)
        theta_init, theta_final = dict(), dict()
        for name, param in basic_model.named_parameters():
            theta_init[name] = nn.parameter.Parameter(3 * torch.rand(param.shape) + 1)
            theta_final[name] = nn.parameter.Parameter(3 * torch.rand(param.shape) + 1)
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_graph(
            alpha_values,
            get_loss(
                model=basic_model,
                alpha_values=alpha_values,
                theta_init=theta_init,
                theta_final=theta_final,
            ),
            label="basic",
        )
        plot_graph(
            alpha_values,
            get_loss(
                model=skip_model,
                alpha_values=alpha_values,
                theta_init=theta_init,
                theta_final=theta_final,
            ),
            label="skip",
        )
        plt.legend()
        plt.show()
