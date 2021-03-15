import sys
import string
import random
from math import sqrt
import time


GA_POPSIZE = 2048		    # ga population size
GA_MAXITER = 16384		    # maximum iterations
GA_ELITRATE = 0.1		    # elitism rate
GA_MUTATIONRATE = 0.25      # mutation rate
GA_MUTATION = sys.maxsize * GA_MUTATIONRATE
HIT_BONUS = 1
EXACT_BONUS = 10
PSO_C1 = 2
PSO_C2 = 2
PSO_W_MIN = 0.4
PSO_W_MAX = 0.9
PSO_LEARNING_RATE = 0.1
GA_TARGET = "Hello World!"


# class of genetic (GA_STRUCT in cpp example)
# compatible with PSO
class Genetic:
    def __init__(self, str_info, pso=False):
        self.str = str_info
        self.fitness = 0
        self.pso = pso
        if pso:
            self.p_best = str_info
            self.p_best_score = sys.maxsize
            self.velocity = [random.randrange(-94, 94) for i in range(len(GA_TARGET))]  # self.velocity = [0 for i in range(len(GA_TARGET))]

    def pso_update(self, best, w):
        for l in range(len(GA_TARGET)):
            ran_p = random.random()
            ran_g = random.random()
            self.velocity[l] = w*self.velocity[l] + PSO_C1*ran_p*(ord(self.p_best[l])-ord(self.str[l])) + PSO_C2*ran_g*(ord(best.str[l])-ord(self.str[l]))

        res = [ord(self.str[i]) + int(PSO_LEARNING_RATE*self.velocity[i]) for i in range(len(GA_TARGET))]
        for i in range(len(res)):
            if res[i] < 32:
                res[i] = 32
            if res[i] > 126:
                res[i] = 126
        self.str = ''.join(chr(i) for i in res)


# EX4: class for "Bul Pgiya" heuristic
class HitBonus:
    def calc_fitness(self, gen_arr):
        for g in gen_arr:
            g.fitness += ((EXACT_BONUS+HIT_BONUS) * len(GA_TARGET))  # !!!!
            for i in range(len(GA_TARGET)):
                if g.str[i] == GA_TARGET[i]:
                    g.fitness -= EXACT_BONUS
                if g.str[i] in GA_TARGET:
                    g.fitness -= HIT_BONUS
            if g.pso and g.p_best_score > g.fitness:
                g.p_best_score = g.fitness
                g.p_best = g.str


# generic letter distance heuristic class
class LetterDistance:
    def calc_fitness(self, gen_arr):
        for g in gen_arr:
            g.fitness = 0
            for i in range(len(GA_TARGET)):
                g.fitness += abs(ord(g.str[i]) - ord(GA_TARGET[i]))
            if g.pso and g.p_best_score > g.fitness:
                g.p_best_score = g.fitness
                g.p_best = g.str


heuristic_dictionary = {'0': LetterDistance(), '1': HitBonus()}


# EX3: classes for each type of crossover requested
# one point crossover class
class OneCross:
    def crossover(self, str1, str2):
        spos = random.randint(0, len(GA_TARGET))
        return str1[:spos] + str2[spos:]


# two point crossover class
class TwoCross:
    def crossover(self, str1, str2):
        spos1 = random.randint(0, len(GA_TARGET))
        spos2 = random.randint(spos1, len(GA_TARGET))
        return str1[:spos1] + str2[spos1:spos2] + str1[spos2:]


# Uniform crossover class
class UniCross:
    def crossover(self, str1, str2):
        mutation = []
        for i in range(len(GA_TARGET)):
            c = random.choice([str1[i], str2[i]])
            mutation.append(c)
        return "".join(mutation)


crossover_dictionary = {'0': UniCross(), '1': OneCross(), '2': TwoCross()}


class RWS:
    def selection(self, gen_arr, k):
        selections = []
        max_fit = sum([(gen_arr[GA_POPSIZE-1].fitness - g.fitness) for g in gen_arr])
        for i in range(k):
            ran_selection = random.uniform(0, max_fit-1)
            current = 0
            for j in range(len(gen_arr)):
                current += gen_arr[j].fitness
                if current > ran_selection:
                    selections.append(j)
                    break
        return selections


class SUS:
    def selection(self, gen_arr, k):
        selections = []
        max_fit = sum([(gen_arr[GA_POPSIZE-1].fitness - g.fitness) for g in gen_arr])
        start_range = 0
        for i in range(k):
            start_range = random.uniform(start_range, max_fit)
            current = 0
            for j in range(len(gen_arr)):
                current += gen_arr[j].fitness
                if current > start_range:
                    selections.append(j)
                    break
        return selections


class TOURNAMENT:
    def selection(self, gen_arr, k):
        pass


class REGULAR:
    def selection(self, gen_arr, k):
        selections = []
        for i in range(k):
            selections.append(random.randint(0, int(GA_POPSIZE/2)))
        return selections


selection_dictionary = {'0': RWS(), '1': SUS(), '2': TOURNAMENT(), '3': REGULAR()}

# creates population
def init_population(pso=False):
    pop = []
    buffer = [Genetic("") for i in range(GA_POPSIZE)]
    for i in range(GA_POPSIZE):
        ran_str = ''.join(chr(random.randrange(32, 126)) for l in range(len(GA_TARGET)))  # ran_str = ''.join(random.choice(string.printable) for l in range(len(GA_TARGET)))   # random string generator
        pop.append(Genetic(ran_str, pso))

    return pop, buffer     # arrays of Genetic type population and buffer initialized


# sorts population by key fitness value
def sort_by_fitness(gen_arr):
    gen_arr.sort(key=lambda x: x.fitness)
    return gen_arr


# takes GA_ELITRATE percent to next generation
def elitism(gen_arr, buffer, esize):
    for i in range(esize):
        buffer[i] = gen_arr[i]
    return buffer


# randomly changes one of the characters
def mutate(buffer, i):
    ipos = random.randint(0, len(GA_TARGET)-1)
    delta = random.choice(string.printable)
    s = list(buffer[i].str)
    s[ipos] = delta
    buffer[i].str = "".join(s)


def mate(gen_arr, buffer, c, selection_type):
    esize = int(GA_POPSIZE * GA_ELITRATE)       # number of elitism moving to next gen
    buffer = elitism(gen_arr, buffer, esize)

    for i in range(GA_POPSIZE-esize):
        index = i+esize
        s = selection_type.selection(gen_arr, 2)
        mut = c.crossover(gen_arr[s[0]].str, gen_arr[s[1]].str)
        buffer[index] = Genetic(mut)

        if random.randint(0, sys.maxsize) < GA_MUTATION:
            mutate(buffer, index)
    return gen_arr, buffer


# calculates average fitness of current generation
def avg_fit(gen_arr):
    totalfit = 0
    for i in range(GA_POPSIZE):
        totalfit += gen_arr[i].fitness
    return totalfit/GA_POPSIZE


# calculates STD of current generation
def std_fit(gen_arr):
    mu = avg_fit(gen_arr)
    squared_dis = 0
    for i in range(GA_POPSIZE):
        squared_dis += (gen_arr[i].fitness-mu)**2
    return sqrt(squared_dis/GA_POPSIZE)


# print function
def print_best(gen_arr):
    print("Best: {} ({}).".format(gen_arr[0].str, gen_arr[0].fitness))
    print("Avg fitness of gen: {}".format(avg_fit(gen_arr)))
    print("Fitness STD: {}".format(std_fit(gen_arr)))


def swap(gen_arr, buffer):
    return buffer, gen_arr


def main():
    cross = crossover_dictionary[input("Choose crossover Type:\n0 - Uniform crossover\n1 - Single point cross over\n2 - Two point cross over\n")]
    heu = heuristic_dictionary[input("Choose heuristic:\n0 - Letter Distance\n1 - Hit Bonus\n")]
    select = selection_dictionary[input("Choose selection:\n0 - RWS\n1 - SUS\n2 - TOURNAMENT\n3 - REGULAR\n")]
    gen_arr, buffer = init_population()
    totaltimer = time.time()
    totalticks = time.process_time()
    for i in range(GA_MAXITER):
        gentimer = time.time()
        genticktimer = time.process_time()
        heu.calc_fitness(gen_arr)
        gen_arr = sort_by_fitness(gen_arr)
        print_best(gen_arr)
        if gen_arr[0].fitness == 0:
            break
        gen_arr, buffer = mate(gen_arr, buffer, cross, select)
        gen_arr, buffer = swap(gen_arr, buffer)
        print("Total time of generation: {}".format(time.time() - gentimer))
        print("Total clock ticks (CPU)) of generation: {}\n".format(time.process_time() - genticktimer))
    print("Total time : {}\nTotal clock ticks : {}\nTotal iter:{}".format(time.time() - totaltimer, time.process_time() - totalticks, i+1))


if __name__ == "__main__":
    main()