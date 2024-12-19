from rich.console import Console
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import torch
from tqdm import tqdm

import copy
import pickle

from tabular.config import Config
from tabular.util import Util
from tabular.skip_nn import SkipNN_1
from tabular.basic_nn import NN_1


def train_one_epoch():
    epoch_loss, total, correct = 0, 0, 0
    num_batches = len(util.train_loader)
    for idx, (data, labels) in enumerate(util.train_loader):
        data, labels = data.to(config.device), labels.to(config.device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # find accuracy
        pred = outputs.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    avg_epoch_loss = epoch_loss / num_batches
    avg_epoch_acc = correct / total
    return avg_epoch_loss, avg_epoch_acc


def eval_one_epoch():
    epoch_loss, total, correct = 0, 0, 0
    num_batches = len(util.test_loader)
    for idx, (data, labels) in enumerate(util.test_loader):
        data, labels = data.to(config.device), labels.to(config.device)
        outputs = model(data)
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    avg_epoch_loss = epoch_loss / num_batches
    avg_epoch_acc = correct / total
    return avg_epoch_loss, avg_epoch_acc


def train_model():
    print()
    train_loss, test_loss, train_acc, test_acc = [], [], [], []
    weights = []
    best_acc, best_model_dict = 0, dict()
    for epoch in tqdm(range(config.num_epochs)):
        model.train(True)
        avg_epoch_loss, avg_epoch_acc = train_one_epoch()
        train_loss.append(avg_epoch_loss)
        train_acc.append(avg_epoch_acc)
        if config.capture_weights:
            weights.append(copy.deepcopy(model.state_dict()))
        print(
            f"Epoch {epoch + 1} / {config.num_epochs} complete, Average Training Loss: {avg_epoch_loss:.4f}, Average Training Accuracy: {avg_epoch_acc:.4f}"
        )

        model.eval()
        with torch.no_grad():
            avg_epoch_loss, avg_epoch_acc = eval_one_epoch()
            test_loss.append(avg_epoch_loss)
            test_acc.append(avg_epoch_acc)
            print(
                f"Epoch {epoch + 1} / {config.num_epochs} complete, Average test Loss: {avg_epoch_loss:.4f}, Average test Accuracy: {avg_epoch_acc:.4f}"
            )
            if avg_epoch_acc > best_acc:
                best_acc = avg_epoch_acc
                best_model_dict = model.state_dict()
    if config.save_model:
        torch.save(best_model_dict, util.get_file_path(file_type=".pth"))
        print("saved best model")
    if config.capture_weights:
        with open(util.get_file_path(file_type=".pkl"), "wb") as weights_file:
            pickle.dump(weights, weights_file)
    return train_loss, test_loss, train_acc, test_acc, best_acc


def plot_graphs():
    x = list(range(len(train_acc)))
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    # accuracy
    ax1.plot(x, train_acc, label="train acc {max(train_acc):.5f} at {train_acc.index(max(train_acc))}")
    ax1.plot(x, test_acc, label="test acc {max(test_acc):.5f} at {test_acc.index(max(test_acc))}")
    ax1.set_xlabel("epoch number")
    ax1.set_ylabel("accuracy")
    ax1.set_title(f"Plot of accuracy vs epoch, best acc = {best_acc}")
    ax1.legend()  # loss
    ax2.plot(x, train_loss, label="train loss {min(train_loss):.5f} at {train_loss.index(min(train_loss))}")
    ax2.plot(x, test_loss, label="test loss {min(test_loss):.5f} at {test_loss.index(min(test_loss))}")
    ax2.set_xlabel("epoch number")
    ax2.set_ylabel("loss")
    ax2.set_title("Plot of loss vs epoch")
    ax2.legend()
    plt.savefig(util.get_file_path(file_type=".png"))


if __name__ == "__main__":
    config = Config()
    util = Util(config)
    print(f'using {config.device}')

    model = globals()[config.model_name](config).to(config.device)
    model.apply(util.init_weights)
    util.plot_architecture(path=util.get_file_path(file_type=""), model=model)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay if config.use_weight_decay else 0,
    )
    criterion = nn.CrossEntropyLoss()
    with open(util.get_file_path(file_type=".txt"), "w") as f:
        with redirect_stdout(f):
            print(model)
            util.write_forward_method(model=model)
            console = Console(file=f)
            util.write_param_count(model=model, console=console)
            if config.do_permutation and "Skip" in config.model_name:
                print(f"permutation = {config.permutation}")
            train_loss, test_loss, train_acc, test_acc, best_acc = train_model()
            plot_graphs()
