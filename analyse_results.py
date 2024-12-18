import matplotlib.pyplot as plt

class Config:
    def __init__(self):
        self.path_1 = r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/skip_connections/runs/0014_SkipNN_1_bsz-128_lr-0.001_ep-200-desc-.txt"
        self.path_2 = r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/skip_connections/runs/0013_SkipNN_1_bsz-128_lr-0.001_ep-200-desc-perm.txt"
        self.path_3 = r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/skip_connections/runs/0012_SkipNN_1_bsz-128_lr-0.001_ep-200-desc-rot-skip-52.txt"
        self.path_4 = r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/skip_connections/runs/0011_SkipNN_1_bsz-128_lr-0.001_ep-200-desc-rot-skip-10.txt"

def get_data(path):
    train_loss, train_acc, test_loss, test_acc = [], [], [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            if line.startswith('Epoch'):
                loss, acc = line.split(',')[1:]
                loss = float(loss.split(':')[1].strip())
                acc = float(acc.split(':')[1].strip())
                if 'Training' in line:
                    train_loss.append(loss)
                    train_acc.append(acc)
                else:
                    test_loss.append(loss)
                    test_acc.append(acc)
    return train_loss, train_acc, test_loss, test_acc

def plot_graphs(train_loss, train_acc, test_loss, test_acc, loss_ax, acc_ax, label, color):
    x = list(range(1, len(train_loss)+1))
    loss_ax.plot(x, train_loss, label=f'{label} train, min={min(train_loss)} at {train_loss.index(min(train_loss))}', color=color)
    # loss_ax.plot(x, test_loss, label=f'{label} test, min={min(test_loss)} at {test_loss.index(min(test_loss))}', linestyle='dashed', color=color)
    # loss_ax.plot(x, test_loss, label=f'{label} test, min={min(test_loss)} at {test_loss.index(min(test_loss))}', color=color)
    loss_ax.set_xlabel('epoch number')
    loss_ax.set_ylabel('loss')
    loss_ax.set_title('Loss Plot')
    acc_ax.plot(x, train_acc, label=f'{label} train, max={max(train_acc)} at {train_acc.index(max(train_acc))}', color=color)
    # acc_ax.plot(x, test_acc, label=f'{label} test, max={max(test_acc)} at {test_acc.index(max(test_acc))}', linestyle='dashed', color=color)
    # acc_ax.plot(x, test_acc, label=f'{label} test, max={max(test_acc)} at {test_acc.index(max(test_acc))}', color=color)
    acc_ax.set_xlabel('epoch number')
    acc_ax.set_ylabel('acc')
    acc_ax.set_title('Accuracy Plot')

    return loss_ax, acc_ax

if __name__ == "__main__":
    config = Config()

    fig, (acc_ax, loss_ax) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    loss_ax, acc_ax = plot_graphs(*get_data(config.path_1), loss_ax, acc_ax, 'skip-direct', color='green')
    loss_ax, acc_ax = plot_graphs(*get_data(config.path_4), loss_ax, acc_ax, 'skip-rot-10', color='blue')
    loss_ax, acc_ax = plot_graphs(*get_data(config.path_2), loss_ax, acc_ax, 'skip-permut', color='red')
    loss_ax, acc_ax = plot_graphs(*get_data(config.path_3), loss_ax, acc_ax, 'skip-rot-52', color='cyan')
    loss_ax.legend()
    acc_ax.legend()
    plt.show()
