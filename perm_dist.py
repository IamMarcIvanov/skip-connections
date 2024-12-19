import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import math

if __name__ == '__main__':
    max_dim = 20
    num_pts_per_dim = 200
    avg_dists = []
    for dim in range(2, max_dim):
        sum_per_dim = 0
        for pt in range(num_pts_per_dim):
            coords = np.zeros((dim,))
            for elem in range(dim):
                coords[elem] = random.uniform(3, 5)
            coords = coords / np.sqrt(coords.square().sum())
            dist_sum = 0
            for perm in permutations(coords):
                dist_sum += np.linalg.norm(coords - np.array(perm))
            sum_per_dim += dist_sum / math.factorial(dim)
        avg_dists.append(sum_per_dim / num_pts_per_dim)
    plt.plot(range(2, max_dim), avg_dists)
    plt.xlabel('dimension')
    plt.ylabel('avergage distance to a permutation')
    plt.show()



