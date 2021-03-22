import GeneticAlgorithm as GA
import matplotlib.pyplot as plt
import numpy as np


REG003 = [23.17, 27.95, 45.66, 89.9, 179.83, 571.02]
t_REG003 = [1.443546, 0.83078, 0.67098, 0.557685, 0.543737, 0.556542]
# mut_cross
# 0_0 ' 1_0 ' 0_1 ' 1_1
suc_RWS = [86, 38, 99, 83]
suc_SUS = [41, 20, 39, 15]
suc_TUR = [1, 2, 94, 90]
suc_REG = [1, 1, 99, 91]

PSO0 = [45.18, 48.9271, 49.7363, 59.2875, 66.74286, 93.53125]
t_PSO0 = [1.602588, 0.80447, 0.4231, 0.206962, 0.109996, 0.049005]
avg_iter_RWS = [38.2907, 45.10526, 18.07071, 19.10843]
avg_iter_SUS = [22.04878, 37.7, 13.69231, 27.26667]
avg_iter_TUR = [59, 49.5, 47.8617, 49.81111]
avg_iter_REG = [74, 92, 45.09091, 42.87912]

REG013 = [15.99, 17.79, 27.69, 78.35, 153.44, 536.58]
t_REG013 = [1.070438, 0.56909, 0.44371, 0.52759, 0.501278, 0.566466]
time_RWS = [28.34724, 36.44966, 12.76833, 13.32319]
time_SUS = [13.77284, 23.11645, 8.003315, 15.00773]
time_TUR = [19.83742, 17.50279, 15.03086, 15.71103]
time_REG = [23.68682, 30.62769, 14.15963, 13.64827]

PSO1 = [61.05682, 67.5309, 73.08, 85.6458, 96.56, 105.5]
t_PSO1 = [2.424585, 1.27757, 0.7017, 0.32896, 0.177291, 0.061538]
# {'0': OXCrossover(), '1': PMXCrossover()}
# {'0': SwapMutation(), '1': ScrambleMutation()}
labels = ['Swap Mutation\nOX Crossover', 'Scramble Mutation\nOX Crossover', 'Swap Mutation\nPMX Crossover', 'Scramble Mutation\nPMX Crossover']

pop = [2000, 1000, 500, 200, 100, 30]
x = np.arange(len(labels))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots(2, 2)
ax[0, 0].set_title('Compare iteration number to population size for both algorithms\nLetters Distance heuristic')
ax[0, 0].set(xlabel='Iteration Number', ylabel='Population Size')
ax[0, 0].plot(REG003, pop)
ax[0, 0].plot(PSO0, pop)
ax[0, 0].legend(["REGULAR", "PSO"])
fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.5*width, time_RWS, width, label='RWS')
rects2 = ax.bar(x - 0.5*width, time_SUS, width, label='SUS')
rects3 = ax.bar(x + 0.5*width, time_TUR, width, label='TOURNAMENT')
rects4 = ax.bar(x + 1.5*width, time_REG, width, label='REGULAR')

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
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Run-Time (sec)')
# ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax[1, 1].set_title('Compare run-time to population size for both algorithms\nHit Bonus heuristic')
ax[1, 1].set(xlabel='Run-time(seconds)', ylabel='Population Size')
ax[1, 1].plot(t_REG013, pop)
ax[1, 1].plot(t_PSO1, pop)
ax[1, 1].legend(["REGULAR", "PSO"])

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = round(rect.get_height(), 2)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()

plt.show()