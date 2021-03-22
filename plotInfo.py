import GeneticAlgorithm as GA
import matplotlib.pyplot as plt
from numpy import average as avg

avg_conflicts = []
avg_iter = []

for i in range(4, 26):
    file = "min_conf/N=" + str(i) + ".txt"
    f = open(file, "r+")
    res = f.read()
    res = res.split()
    iter = [float(res[i]) for i in range(len(res)) if not i % 2]
    conflics = [float(res[i]) for i in range(len(res)) if i % 2]
    avg_iter.append(avg(iter))
    avg_conflicts.append(avg(conflics))

fig, ax = plt.subplots(2)
ax[0].set_title('Compare size of board to number of iterations to find local minimum on average')
ax[0].set(xlabel='Board Size', ylabel='Iteration Number (avg)')
ax[0].plot(list(range(4,26)), avg_iter)

ax[1].set_title('Compare size of board to number of conflict in best board found (local minimum) on average')
ax[1].set(xlabel='Board Size', ylabel='Number of Conflicts (avg)')
ax[1].plot(list(range(4, 26)), avg_conflicts)



plt.show()
