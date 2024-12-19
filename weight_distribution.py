import matplotlib.pyplot as plt
import numpy as np
from rich.pretty import pprint
from scipy.stats import norm
import torch
from tqdm import tqdm

import pickle


def get_norm_seq(state_dict_seq):
    wt_norms = []
    for state_dict in state_dict_seq:
        all_weights_sum_sq = 0
        for w in state_dict.values():
            all_weights_sum_sq += w.detach().cpu().square().sum()
        wt_norms.append(torch.sqrt(all_weights_sum_sq))
    return wt_norms


def plot_norms(norms, labels, colors):
    fig, ax = plt.subplots(figsize=(10, 6))
    for norm, label, color in zip(norms, labels, colors):
        ax.plot(range(len(norm)), norm, label=label, color=color)
    ax.legend()
    ax.set_xlabel("epoch")
    ax.set_ylabel("L2 norm")
    ax.set_title("basic NN, Adam, runs 18, 19")
    fig.savefig('./results/l2_norm_basic_nn_runs_dec_5e-4_26_27_with_decay.png')
    plt.show()


def plot_wt_dist(state_dicts, labels, colors):
    fig, ax = plt.subplots(figsize=(10, 6))
    for state_dict, label, color in zip(state_dicts, labels, colors):
        res = torch.Tensor()
        for w in state_dict.values():
            res = torch.cat([res, torch.ravel(w.detach().cpu())])
        res_np = res.numpy()
        # mean, std = res.mean(), res.std()
        counts, bins = np.histogram(res_np, bins=100)
        ax.hist(bins[:-1], bins, weights=counts, label=label, color=color, alpha=0.5)
        # x = np.linspace(min(bins), max(bins), 2000)
        # ax.plot(x, norm.pdf(x, mean, std), label=f'{label} norm', color=color, linestyle='dashed')
    ax.legend()
    ax.set_xlabel("weight")
    ax.set_ylabel("frequency")
    ax.set_title("basic NN, Adam, run 18, 19, 100 epochs, decay=0")
    fig.savefig('./results/wt_dist_bsz_128_8192_ep_100_basic_nn_dec-0_runs_28_29.png')
    plt.show()

def print_weight(state_dicts):
    norm = []
    for state_dict in state_dicts:
        w = state_dict['fc_1.bias'].detach().cpu()
        norm.append(torch.sqrt(w.square().sum()).item())
    plt.plot(range(len(norm)), norm)
    plt.show()
    # false_count = 0
    # for i in tqdm(range(50000)):
    #     val_1 = np.random.randint(low=0, high=len(state_dicts))
    #     val_2 = np.random.randint(low=0, high=len(state_dicts))
    #     if not (state_dicts[val_1]['fc_1.bias'] == state_dicts[val_2]['fc_1.bias']).all():
    #         false_count += 1
    # print(false_count)


if __name__ == "__main__":
    # wt_path_2 = r"./runs/0027_NN_1_bsz-8192_lr-0.001_ep-100-desc-wt-dec-5e-4.pkl"
    # wt_path_1 = r"./runs/0026_NN_1_bsz-128_lr-0.001_ep-100-desc-wt-dec-5e-4.pkl"
    wt_path_2 = r"./runs/0028_NN_1_bsz-8192_lr-0.001_ep-100-desc-wt-dec-0.pkl"
    wt_path_1 = r"./runs/0029_NN_1_bsz-128_lr-0.001_ep-100-desc-wt-dec-0.pkl"
    with open(wt_path_1, "rb") as f:
        wts_1 = pickle.load(f)
    with open(wt_path_2, "rb") as f:
        wts_2 = pickle.load(f)
    # print_weight(wts_1)
    wt_norms_1 = get_norm_seq(wts_1)
    wt_norms_2 = get_norm_seq(wts_2)
    # plot_norms([wt_norms_1, wt_norms_2], ["128", "8192"], ["red", "blue"])
    plot_wt_dist([wts_1[-1], wts_2[-1]], ["128", "8192"], ["red", "blue"])
