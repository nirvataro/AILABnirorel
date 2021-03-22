import GeneticAlgorithm as GA
import time
import random
from psutil import cpu_freq


MAXITER = 500
POPSIZE = 1000

MAX_WEIGHT = 6404180
ITEM_PRICE = [825594, 1677009, 1676628, 1523970, 943972, 97426, 69666, 1296457, 1679693, 1902996, 1844992, 1049289, 1252836, 1319836, 953277, 2067538, 675367, 853655, 1826027, 65731, 901489, 577243, 466257, 369261]
SOLUTION = [1,1,0,1,1,1,0,0,0,1,1,0,1,0,0,1,0,0,0,0,0,1,1,1]
ITEM_WEIGHT = [382745,799601,909247,729069,467902,44328,34610,698150,823460,903959,853665,551830,610856,670702,488960,951111,323046,446298,931161,31385,496951,264724,224916,169684]


def generateBag():
    bag = [0 for i in range(len(SOLUTION))]
    for i in range(len(SOLUTION)):
        bag[i] = random.choice([0, 1])
    return bag


def init():
    pop, buffer = [], []
    for i in range(POPSIZE):
        ran_str = generateBag()
        pop.append(GA.Genetic(ran_str))
        buffer.append(GA.Genetic(ran_str))

    return pop, buffer     # arrays of Genetic type population and buffer initialized


class Knapsack:
    def calc_personal_fitness(self, bag, tar_len):
        total_price = sum(ITEM_PRICE)
        fitness = total_price
        total_weight = 0
        for i in range(len(bag.str)):
            if bag.str[i]:
                fitness -= ITEM_PRICE[i]
                total_weight += ITEM_WEIGHT[i]
        overweight = total_weight - MAX_WEIGHT
        if overweight > 0:
            fitness += overweight*total_price
        return fitness


def main():
    gen_arr, buffer = init()
    ks = Knapsack()

    total_timer = time.time()

    for i in range(MAXITER):
        gen_timer = time.time()

        GA.calc_fitness(gen_arr, ks, tar_len=len(SOLUTION))
        gen_arr = GA.sort_by_fitness(gen_arr)
        GA.print_best(gen_arr[0], gen_arr, gen_timer)
        if gen_arr[0].str == SOLUTION:
            total_time = time.time() - total_timer
            print("Total time : {}\nTotal clock ticks : {}\nTotal iter:{}".format(total_time,
                                                                                  total_time * cpu_freq()[0] * 2 ** 20,
                                                                                  i + 1))
            return True
            # break

        gen_arr = GA.birthday(gen_arr)
        # mate and swap between buffer and gen_arr
        buffer, gen_arr = GA.mate(gen_arr, buffer, GA.OneCross(), GA.RWS(), init_values=[0, 1], min_age=3, tar_len=len(SOLUTION), flag_Knapsack=True)

    total_time = time.time() - total_timer
    print("Total time : {}\nTotal clock ticks : {}\nTotal iter:{}".format(total_time, total_time*cpu_freq()[0]*2**20, i+1))
    return False


if __name__ == "__main__":
    suc = False
    tries = 0
    while not suc:
        print(tries)
        suc = main()
        tries += 1


