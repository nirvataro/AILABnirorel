import GeneticAlgorithm as GA
import matplotlib.pyplot as plt


REG003 = [23.17, 27.95, 45.66, 89.9, 179.83, 571.02]
t_REG003 = [1.443546, 0.83078, 0.67098, 0.557685, 0.543737, 0.556542]

PSO0 = [45.18, 48.9271, 49.7363, 59.2875, 66.74286, 93.53125]
t_PSO0 = [1.602588, 0.80447, 0.4231, 0.206962, 0.109996, 0.049005]

REG013 = [15.99, 17.79, 27.69, 78.35, 153.44, 536.58]
t_REG013 = [1.070438, 0.56909, 0.44371, 0.52759, 0.501278, 0.566466]

PSO1 = [61.05682, 67.5309, 73.08, 85.6458, 96.56, 105.5]
t_PSO1 = [2.424585, 1.27757, 0.7017, 0.32896, 0.177291, 0.061538]

pop = [2000, 1000, 500, 200, 100, 30]

fig, ax = plt.subplots(2, 2)
ax[0, 0].set_title('Compare iteration number to population size for both algorithms\nLetters Distance heuristic')
ax[0, 0].set(xlabel='Iteration Number', ylabel='Population Size')
ax[0, 0].plot(REG003, pop)
ax[0, 0].plot(PSO0, pop)
ax[0, 0].legend(["REGULAR", "PSO"])

ax[1, 0].set_title('Compare run-time to population size for both algorithms\nLetters Distance heuristic')
ax[1, 0].set(xlabel='Run-time(seconds)', ylabel='Population Size')
ax[1, 0].plot(t_REG003, pop)
ax[1, 0].plot(t_PSO0, pop)
ax[1, 0].legend(["REGULAR", "PSO"])

ax[0, 1].set_title('Compare iteration number to population size for both algorithms\nHit Bonus heuristic')
ax[0, 1].set(xlabel='Iteration Number', ylabel='Population Size')
ax[0, 1].plot(REG013, pop)
ax[0, 1].plot(PSO1, pop)
ax[0, 1].legend(["REGULAR", "PSO"])

ax[1, 1].set_title('Compare run-time to population size for both algorithms\nHit Bonus heuristic')
ax[1, 1].set(xlabel='Run-time(seconds)', ylabel='Population Size')
ax[1, 1].plot(t_REG013, pop)
ax[1, 1].plot(t_PSO1, pop)
ax[1, 1].legend(["REGULAR", "PSO"])

plt.show()
