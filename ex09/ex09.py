import numpy as np
import itertools as it

loadtxt = np.loadtxt("data/model_forecasts.txt", delimiter=",")

teams = np.arange(10)
all_perms = it.permutations(teams)
all_perms = np.array(list(all_perms))
team1 = all_perms[:, 0::2]
team2 = all_perms[:, 1::2]

diff = loadtxt[team1, team2]

sum_sq = np.sum(diff ** 2, axis=1)
 
best_idx  = np.argmin(sum_sq)
best_perm = all_perms[best_idx]
 
result = np.array([best_perm[0::2],best_perm[1::2]])
 
print(result)
