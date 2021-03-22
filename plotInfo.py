import GeneticAlgorithm as GA
import matplotlib.pyplot as plt


pop_size = [200, 500, 1000, 2000]
success = [37, 49, 56, 74]
seconds = [7.295, 14.836, 32.827, 126.315]
iterations = [41.621, 37.458, 30.2, 38.917]

fig, ax = plt.subplots(3)
# ax[0, 0].set_title('Compare iteration number to population size for both algorithms\nLetters Distance heuristic')
ax[0].set(xlabel='Population Size', ylabel='Success Rate(%)')
ax[0].plot(pop_size, success)
# ax[0].plot(, pop)

#ax[1, 0].set_title('Compare run-time to population size for both algorithms\nLetters Distance heuristic')
ax[1].set(xlabel='Population Size', ylabel='Run-Time (sec)')
ax[1].plot(pop_size, seconds)
# ax[1].plot(t_PSO0, pop)
#ax[1].legend(["REGULAR", "PSO"])

#ax[2].set_title('Compare iteration number to population size for both algorithms\nHit Bonus heuristic')
ax[2].set(xlabel='Population Size', ylabel='Iteration Number')
ax[2].plot(pop_size, iterations)
#ax[2].plot(PSO1, pop)
#ax[2].legend(["REGULAR", "PSO"])

# ax[1, 1].set_title('Compare run-time to population size for both algorithms\nHit Bonus heuristic')
# ax[1, 1].set(xlabel='Run-time(seconds)', ylabel='Population Size')
# ax[1, 1].plot(t_REG013, pop)
# ax[1, 1].plot(t_PSO1, pop)
# ax[1, 1].legend(["REGULAR", "PSO"])

plt.show()
