import os
import math
import torch
import inspect
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from rich.table import Table
from torchviz import make_dot
import matplotlib.pyplot as plt
from rich.console import Console
from torchvision import datasets
from torchvision import transforms
from contextlib import redirect_stdout
from torch.utils.data import Dataset, DataLoader


class Config:
    def __init__(self):
        self.data_dir = r"/mnt/windows/Users/lordh/Documents/Svalbard/Data/skip_connections/cifar_10/"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 128
        self.lr = 0.001
        self.num_epochs = 60
        self.model_name = "CNN_2"
        self.description = ""
        self.save_model = True
        self.results_dir = r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/skip_connections/runs/"
        self.results_file = r"{}_{}_bsz-{}_lr-{}_ep-{}-desc-{}{}"


class Util:
    def __init__(self, config):
        self.config = config
        self.train_set, self.test_set, self.train_loader, self.test_loader = (
            None,
            None,
            None,
            None,
        )
        self.num_classes = 10
        self.file_num = self.get_file_num()

        self.set_data()

    def set_data(self):
        # for Celeb A
        # in the train set there are 162,770 images
        # the image size is 3 x 218 x 178
        # the number of labels is 10,177 - they go from 1 to 10,177
        # the number of labels per class is around 30, majority of them have 30,
        # with some having upto 35 and others upto 22
        # for CIFAR 10
        # 50k train images, 10k test images
        # image size = 3 x 32 x 32
        # there are 10 classes, with 6k images per class
        train_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                # for CIFAR the normalisation does not seem necessary
                # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
            ]
        )
        self.train_set = datasets.CIFAR10(
            root=self.config.data_dir,
            train=True,
            download=True,
            transform=train_transforms,
        )
        self.test_set = datasets.CIFAR10(
            root=self.config.data_dir,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        self.train_loader = DataLoader(
            self.train_set, batch_size=config.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_set, batch_size=config.batch_size, shuffle=False
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


class SkipCNN_2(nn.Module):
    def __init__(self, config, util):
        super().__init__()
        # conv 1 and activation 1
        # this conv_1 + pool_1 decreases the input to N, 32, 14, 14
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.act_1 = nn.ReLU()
        h_out = util.get_l_out(l_in=32, kernel_size=3)
        w_out = util.get_l_out(l_in=32, kernel_size=3)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        h_out = util.get_l_out(l_in=h_out, kernel_size=3, stride=2)
        w_out = util.get_l_out(l_in=w_out, kernel_size=3, stride=2)

        # conv 2 and act 2
        # conv_2 preserves the dimension of its input
        self.conv_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.batch_norm_2 = nn.BatchNorm2d(32)
        self.act_2 = nn.ReLU()
        h_out = util.get_l_out(l_in=h_out, kernel_size=5, stride=1, padding=2)
        w_out = util.get_l_out(l_in=w_out, kernel_size=5, stride=1, padding=2)

        # conv_3 and act_3 - this gets the input in the resnet
        # this does not have to preserve the dimensionality
        self.conv_3 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            stride=1,
        )
        self.batch_norm_3 = nn.BatchNorm2d(32)
        self.act_3 = nn.ReLU()
        h_out = util.get_l_out(l_in=h_out, kernel_size=5, stride=1)
        w_out = util.get_l_out(l_in=w_out, kernel_size=5, stride=1)

        # max pool 2
        self.max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2)
        h_out = util.get_l_out(l_in=h_out, kernel_size=3, stride=2)
        w_out = util.get_l_out(l_in=w_out, kernel_size=3, stride=2)

        # output layer
        in_features = h_out * w_out * 32
        self.flatten = nn.Flatten()
        self.output = nn.Linear(in_features=in_features, out_features=util.num_classes)

    def forward(self, x):
        # conv block 1
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.act_1(x)
        x1 = self.max_pool_1(x)
        # conv block 2
        x = self.conv_2(x1)
        x = self.batch_norm_2(x1)
        x = self.act_2(x1)
        # conv block 3
        x = self.conv_3(x1 + x)
        x = self.batch_norm_3(x)
        x = self.act_3(x)
        x = self.max_pool_2(x)

        # FC layers
        x = self.flatten(x)
        x = self.output(x)
        return x


class CNN_2(nn.Module):
    def __init__(self, config, util):
        super().__init__()
        # conv 1 and activation 1
        # this conv_1 + pool_1 decreases the input to N, 32, 14, 14
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.act_1 = nn.ReLU()
        h_out = util.get_l_out(l_in=32, kernel_size=3)
        w_out = util.get_l_out(l_in=32, kernel_size=3)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        h_out = util.get_l_out(l_in=h_out, kernel_size=3, stride=2)
        w_out = util.get_l_out(l_in=w_out, kernel_size=3, stride=2)

        # conv 2 and act 2
        # conv_2 preserves the dimension of its input
        self.conv_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.batch_norm_2 = nn.BatchNorm2d(32)
        self.act_2 = nn.ReLU()
        h_out = util.get_l_out(l_in=h_out, kernel_size=5, stride=1, padding=2)
        w_out = util.get_l_out(l_in=w_out, kernel_size=5, stride=1, padding=2)

        # conv_3 and act_3 - this gets the input in the resnet
        # this does not have to preserve the dimensionality
        self.conv_3 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            stride=1,
        )
        self.batch_norm_3 = nn.BatchNorm2d(32)
        self.act_3 = nn.ReLU()
        h_out = util.get_l_out(l_in=h_out, kernel_size=5, stride=1)
        w_out = util.get_l_out(l_in=w_out, kernel_size=5, stride=1)

        # max pool 2
        self.max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2)
        h_out = util.get_l_out(l_in=h_out, kernel_size=3, stride=2)
        w_out = util.get_l_out(l_in=w_out, kernel_size=3, stride=2)

        # output layer
        in_features = h_out * w_out * 32
        self.flatten = nn.Flatten()
        self.output = nn.Linear(in_features=in_features, out_features=util.num_classes)

    def forward(self, x):
        # conv block 1
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.act_1(x)
        x = self.max_pool_1(x)
        # conv block 2
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.act_2(x)
        # conv block 3
        x = self.conv_3(x)
        x = self.batch_norm_3(x)
        x = self.act_3(x)
        x = self.max_pool_2(x)

        # FC layers
        x = self.flatten(x)
        x = self.output(x)
        return x


class CNN_1(nn.Module):
    def __init__(self, config, util):
        super().__init__()
        # conv 1 and activation 1
        # padding is kept to (kernel_size - 1) // 2 to ensure that out size = in size
        self.conv_1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=5, padding=2
        )
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.act_1 = nn.ReLU()
        h_out = util.get_l_out(l_in=32, kernel_size=5, padding=2)
        w_out = util.get_l_out(l_in=32, kernel_size=5, padding=2)

        # conv 2 and act 2
        self.conv_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1
        )
        self.batch_norm_2 = nn.BatchNorm2d(32)
        self.act_2 = nn.ReLU()
        h_out = util.get_l_out(l_in=h_out, kernel_size=5, stride=1)
        w_out = util.get_l_out(l_in=w_out, kernel_size=5, stride=1)

        # max pool 2
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        h_out = util.get_l_out(l_in=h_out, kernel_size=2, stride=2)
        w_out = util.get_l_out(l_in=w_out, kernel_size=2, stride=2)

        # output layer
        in_features = h_out * w_out * 32
        self.flatten = nn.Flatten()
        self.output = nn.Linear(in_features=in_features, out_features=util.num_classes)

    def forward(self, x):
        # conv block 1
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.act_1(x)
        # conv block 2
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.act_2(x)
        x = self.max_pool_2(x)

        # FC layers
        x = self.flatten(x)
        x = self.output(x)
        return x


def check_image_size_and_labels(config, util):
    num_images = len(util.train_set)
    img_lengths, img_widths, img_labels = set(), set(), set()
    for idx in tqdm(range(num_images)):
        img, label = util.train_set[idx]
        _, img_len, img_wid = img.shape
        img_lengths.add(img_len)
        img_widths.add(img_wid)
        img_labels.add(label.item())
    print(img_lengths)
    print(img_widths)
    print(img_labels)


def check_per_class_instances(config, util):
    num_images = len(util.train_set)
    label_count = [0] * (util.num_classes + 1)
    for idx in tqdm(range(num_images)):
        img, label = util.train_set[idx]
        label_count[label.item()] += 1
    plt.plot(list(range(1, util.num_classes + 1)), label_count[1:])
    plt.show()


def plot_pixel_dist(config, util):
    # this was done on non-nomalised and untransformed data
    image, label = util.train_set[3]
    data = image[0].ravel().numpy()
    counts, bins = np.histogram(data, bins=40)
    plt.hist(bins[:-1], bins, weights=counts, label="r")
    data = image[1].ravel().numpy()
    counts, bins = np.histogram(data, bins=40)
    plt.hist(bins[:-1], bins, weights=counts, label="g")
    data = image[2].ravel().numpy()
    counts, bins = np.histogram(data, bins=40)
    plt.hist(bins[:-1], bins, weights=counts, label="b")
    plt.xlabel("pixel values")
    plt.ylabel("freq")
    plt.title("pixel value distribution")
    plt.show()


def check_means_and_std(config, util):
    # this is before any transform - to find the mean and std to use
    num_images = len(util.train_set)
    means_r, means_g, means_b = [], [], []
    std_r, std_g, std_b = [], [], []
    for idx in tqdm(range(num_images)):
        image, label = util.train_set[idx]
        std, mean = torch.std_mean(image[0].ravel())
        std_r.append(std)
        means_r.append(mean)
        std, mean = torch.std_mean(image[1].ravel())
        std_g.append(std)
        means_g.append(mean)
        std, mean = torch.std_mean(image[2].ravel())
        std_b.append(std)
        means_b.append(mean)
    counts, bins = np.histogram(np.array(means_r), bins=40)
    plt.hist(bins[:-1], bins, weights=counts, label="r")
    counts, bins = np.histogram(np.array(means_g), bins=40)
    plt.hist(bins[:-1], bins, weights=counts, label="g")
    counts, bins = np.histogram(np.array(means_b), bins=40)
    plt.hist(bins[:-1], bins, weights=counts, label="b")
    plt.xlabel("values")
    plt.ylabel("freq")
    plt.legend()
    plt.title("dist of means")
    plt.show()
    counts, bins = np.histogram(np.array(std_r), bins=40)
    plt.hist(bins[:-1], bins, weights=counts, label="r")
    counts, bins = np.histogram(np.array(std_g), bins=40)
    plt.hist(bins[:-1], bins, weights=counts, label="g")
    counts, bins = np.histogram(np.array(std_b), bins=40)
    plt.hist(bins[:-1], bins, weights=counts, label="b")
    plt.xlabel("values")
    plt.ylabel("freq")
    plt.legend()
    plt.title("dist of std")
    plt.show()


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
    best_acc, best_model_dict = 0, dict()
    for epoch in tqdm(range(config.num_epochs)):
        model.train(True)
        avg_epoch_loss, avg_epoch_acc = train_one_epoch()
        train_loss.append(avg_epoch_loss)
        train_acc.append(avg_epoch_acc)
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
    return train_loss, test_loss, train_acc, test_acc, best_acc


def plot_graphs():
    x = list(range(len(train_acc)))
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    # accuracy
    ax1.plot(x, train_acc, label="training accuracy")
    ax1.plot(x, test_acc, label="test accuracy")
    ax1.set_xlabel("epoch number")
    ax1.set_ylabel("accuracy")
    ax1.set_title(f"Plot of accuracy vs epoch, best acc = {best_acc}")
    ax1.legend()  # loss
    ax2.plot(x, train_loss, label="training loss")
    ax2.plot(x, test_loss, label="test loss")
    ax2.set_xlabel("epoch number")
    ax2.set_ylabel("loss")
    ax2.set_title("Plot of loss vs epoch")
    ax2.legend()
    plt.savefig(util.get_file_path(file_type=".png"))


def write_forward_method():
    print(inspect.getsource(model.forward))


def plot_architecture(path):
    random_input = torch.rand(1, 3, 32, 32).to(config.device)
    output = model(random_input)
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.render(filename=path)


def write_param_count():
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


if __name__ == "__main__":
    config = Config()
    util = Util(config)

    model = globals()[config.model_name](config, util).to(config.device)
    model.apply(util.init_weights)
    plot_architecture(util.get_file_path(file_type=""))
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    with open(util.get_file_path(file_type=".txt"), "w") as f:
        with redirect_stdout(f):
            print(model)
            write_forward_method()
            console = Console(file=f)
            write_param_count()
            train_loss, test_loss, train_acc, test_acc, best_acc = train_model()
            plot_graphs()
