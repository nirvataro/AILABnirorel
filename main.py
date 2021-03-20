import GeneticAlgorithm as GA
import time
import copy

MAXITER = 2000
POPSIZE = 2000
MAX_WEIGHT = 165
ITEM_PRICE = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
SOLUTION = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
ITEM_WEIGHT = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]


def generateBag(item_weight):
    sum = 0
    bag = [0 for i in range(len(item_weight))]
    indices = list(range(len(item_weight)))
    while sum < MAX_WEIGHT:
        index = GA.random.choice(indices)
        bag[index] = 1
        sum += item_weight[index]
        indices.remove(index)
    bag[index] = 0
    return bag


def init(item_weight):
    pop, buffer = [], []
    for i in range(POPSIZE):
        ran_str = generateBag(item_weight)
        pop.append(GA.Genetic(ran_str))
        buffer.append(GA.Genetic(ran_str))

    return pop, buffer     # arrays of Genetic type population and buffer initialized


class Knapsack:
    def calc_personal_fitness(self, bag):
        fitness = MAX_WEIGHT
        for i in range(len(bag.str)):
            if not bag.str[i]:
                fitness += ITEM_PRICE[i] - ITEM_WEIGHT[i]
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
        buffer, gen_arr = GA.mate(gen_arr, buffer, GA.UniCross(), GA.REGULAR())

    print("Total time : {}\nTotal clock ticks : {}\nTotal iter:{}".format(time.time() - totaltimer, time.process_time() - totalticks, i+1))


if __name__ == "__main__":
    main()
