import GeneticAlgorithm as GA
import time
import random

MAXITER = 2000
POPSIZE = 20
MAX_WEIGHT = 165
ITEM_PRICE = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
SOLUTION = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
ITEM_WEIGHT = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]
PENALTY = 10


def generateBag():
    bag = [None for i in range(len(SOLUTION))]
    for i in range(len(SOLUTION)):
        bag[i] = random.choice([0, 1])
    return bag


def init(item_weight):
    pop, buffer = [], []
    for i in range(POPSIZE):
        ran_str = generateBag()
        pop.append(GA.Genetic(ran_str))
        buffer.append(GA.Genetic(ran_str))

    return pop, buffer     # arrays of Genetic type population and buffer initialized


class Knapsack:
    def calc_personal_fitness(self, bag):
        fitness = sum(ITEM_PRICE)
        total_weight = 0
        for i in range(len(bag.str)):
            if bag.str[i]:
                fitness -= ITEM_PRICE[i]
                total_weight += ITEM_WEIGHT[i]
        overweight = total_weight - MAX_WEIGHT
        if overweight > 0:
            fitness += overweight*PENALTY
        return fitness


def main():
    gen_arr, buffer = init(ITEM_WEIGHT)
    ks = Knapsack()

    totaltimer = time.time()
    totalticks = time.process_time()

    for i in range(MAXITER):
        gentimer = time.time()
        genticktimer = time.process_time()

        GA.calc_fitness(gen_arr, ks)
        gen_arr = GA.sort_by_fitness(gen_arr)
        GA.print_best(gen_arr[0], gen_arr, gentimer, genticktimer)
        if gen_arr[0].str == SOLUTION:
            break

        gen_arr = GA.birthday(gen_arr)
        # mate and swap between buffer and gen_arr
        buffer, gen_arr = GA.mate(gen_arr, buffer, GA.UniCross(), GA.REGULAR(), [0,1], tar_len=len(SOLUTION))

    print("Total time : {}\nTotal clock ticks : {}\nTotal iter:{}".format(time.time() - totaltimer, time.process_time() - totalticks, i+1))


if __name__ == "__main__":
    main()
