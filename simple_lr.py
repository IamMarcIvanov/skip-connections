import matplotlib.pyplot as plt
import numpy as np


def loss(wt):
    return pow(10,4) * ((wt - 3) ** 2)


def main():
    wts = [wt_init]
    wt = wt_init
    for step in range(num_steps):
        wt = wt - alpha * pow(10, 4) * 0.5 * (wt - 3)
        wts.append(wt)
    x = np.linspace(0, 6, 200)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    ax1.plot(x, np.vectorize(loss)(x))
    ax1.scatter(wts, np.vectorize(loss)(np.array(wts)), color='red', s=5)
    ax2.plot(range(num_steps+1), np.vectorize(loss)(np.array(wts)))
    plt.show()


if __name__ == "__main__":
    num_steps = 200
    wt_init = 7
    alpha = 1.5
    main()
