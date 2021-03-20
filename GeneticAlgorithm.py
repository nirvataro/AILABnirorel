import sys
import string
import random
import time
import numpy as np


GA_POPSIZE = 30	    # ga population size
GA_MAXITER = 2000   	    # maximum iterations
GA_ELITRATE = 0.1		    # elitism rate
GA_MUTATIONRATE = 0.25      # mutation rate
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
        self.age = 0
        if pso:
            self.p_best_str = str_info
            self.p_best_fitness = sys.maxsize
            self.velocity = [0 for i in range(len(GA_TARGET))]

    def pso_update(self, best, w, heu):
        self.velocity = [x * w for x in self.velocity]
        for l in range(len(GA_TARGET)):
            const_p = PSO_C1*random.random()
            const_g = PSO_C2*random.random()
            self.velocity[l] += const_p*(ord(self.p_best_str[l])-ord(self.str[l])) + const_g*(ord(best.str[l])-ord(self.str[l]))

        res = [ord(self.str[i]) + round(PSO_LEARNING_RATE*self.velocity[i]) for i in range(len(GA_TARGET))]
        for i in range(len(res)):
            if res[i] < 0:
                res[i] = 0
            if res[i] > 126:
                 res[i] = 126
        self.str = ''.join(chr(i) for i in res)
        self.fitness = heu.calc_personal_fitness(self)
        if self.fitness < self.p_best_fitness:
            self.p_best_fitness = self.fitness
            self.p_best_str = self.str


def calc_fitness(gen_arr, heu):
    for g in gen_arr:
        g.fitness = heu.calc_personal_fitness(g)
        if g.pso and g.p_best_fitness > g.fitness:
            g.p_best_fitness = g.fitness
            g.p_best_str = g.str


# EX4: class for "Bul Pgiya" heuristic
class HitBonus:
    def calc_personal_fitness(self, gen):
        LDcalc = LetterDistance()
        temp_fit = LDcalc.calc_personal_fitness(gen)
        temp_fit += ((EXACT_BONUS + HIT_BONUS) * len(GA_TARGET))
        for i in range(len(GA_TARGET)):
            if gen.str[i] == GA_TARGET[i]:
                temp_fit -= EXACT_BONUS
            if gen.str[i] in GA_TARGET:
                temp_fit -= HIT_BONUS
        return temp_fit


# generic letter distance heuristic class
class LetterDistance:
    def calc_personal_fitness(self, gen):
        fitness = 0
        for i in range(len(GA_TARGET)):
            fitness += abs(ord(gen.str[i]) - ord(GA_TARGET[i]))
        return fitness


heuristic_dictionary = {0: LetterDistance(), 1: HitBonus()}


# EX3: classes for each type of crossover requested
# one point crossover class
class OneCross:
    def crossover(self, str1, str2):
        pos = random.randint(0, len(GA_TARGET))
        return "".join(str1[:pos] + str2[pos:])


# two point crossover class
class TwoCross:
    def crossover(self, str1, str2):
        pos1 = random.randint(0, len(GA_TARGET))
        pos2 = random.randint(pos1, len(GA_TARGET))
        return "".join(str1[:pos1] + str2[pos1:pos2] + str1[pos2:])


# Uniform crossover class
class UniCross:
    def crossover(self, str1, str2):
        mutation = "".join([random.choice([str1[i], str2[i]]) for i in range(len(GA_TARGET))])
        return mutation


crossover_dictionary = {0: UniCross(), 1: OneCross(), 2: TwoCross()}


def scale(gen_arr):
    scaled_fit = [None for i in range(len(gen_arr))]
    for i in range(len(gen_arr)):
        scaled_fit[i] = 1/gen_arr[i].fitness
    return sum(scaled_fit), scaled_fit


class RWS:
    def selection(self, gen_arr, parent=2):
        sel = [None for i in range(parent)]
        total_fitness, scaled_fit = scale(gen_arr)
        for i in range(parent):
            ran_selection = random.uniform(0, total_fitness)
            current, j = 0, 0
            while current < ran_selection:
                current += gen_arr[j].fitness
                j += 1
            sel[i] = gen_arr[j]
        return sel


class SUS:
    def selection(self, gen_arr, parent=2):
        sel = [None for i in range(parent)]
        total_fitness, scaled_fit = scale(gen_arr)
        ran = random.uniform(0, total_fitness/GA_POPSIZE)
        delta = total_fitness/parent
        for i in range(parent):
            fitness = ran + i*delta
            current, j = 0, 0
            while current < fitness:
                current += gen_arr[j].fitness
                j += 1
            sel[i] = gen_arr[j]
        return sel


class TOURNAMENT:
    def selection(self, gen_arr, k_tour_size, parent=2):
        sel = [None for i in range(parent)]
        for i in range(parent):
            tournament = random.choices(gen_arr, k=k_tour_size)
            sel[i] = min(tournament, key=lambda x: x.fitness)
        return sel


class REGULAR:
    def selection(self, gen_arr, k):
        return [gen_arr[random.randint(0, int(GA_POPSIZE/2))] for i in range(k)]


selection_dictionary = {0: RWS(), 1: SUS(), 2: TOURNAMENT(), 3: REGULAR()}


# creates population
def init_population(values, pso=False):
    pop, buffer = [], []
    for i in range(GA_POPSIZE):
        ran_str = ''.join(chr(random.choice(values)) for l in range(len(GA_TARGET)))  # random string generator
        pop.append(Genetic(ran_str, pso))
        buffer.append(Genetic(ran_str, pso))
    if pso:
        return pop

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


def ageing(gen_arr, min_age):
    can_mate = []
    for g in gen_arr:
        if g.age >= min_age:
            can_mate.append(g)
    return can_mate


# randomly changes one of the characters
class RandomMutate:
    def mutate(self, gen):
        pos = random.randint(0, len(GA_TARGET)-1)
        delta = random.choice(string.printable)
        s = list(gen.str)
        s[pos] = delta
        gen.str = "".join(s)


def mate(gen_arr, buffer, crossover_type, selection_type, mut_method=RandomMutate(), min_age=0):
    esize = int(GA_POPSIZE * GA_ELITRATE)       # number of elitism moving to next gen
    buffer = elitism(gen_arr, buffer, esize)
    can_mate = ageing(gen_arr, min_age)

    if len(can_mate) > 0:
        for i in range(esize, GA_POPSIZE):
            s = selection_type.selection(gen_arr, 2)
            mut = crossover_type.crossover(s[0].str, s[1].str)
            buffer[i] = Genetic(mut)

            if np.random.choice([True, False], p=[GA_MUTATIONRATE, 1-GA_MUTATIONRATE]):
                mut_method.mutate(buffer[i])
    return gen_arr, buffer


# calculates average fitness of current generation
def avg_fit(gen_arr):
    fit_arr = [g.fitness for g in gen_arr]
    return np.mean(fit_arr)


# calculates STD of current generation
def std_fit(gen_arr):
    fit_arr = [g.fitness for g in gen_arr]
    return np.std(fit_arr)


# print function
def print_best(best, gen_arr, timer, ticks):
    print("Best: {} ({}).".format(best.str, best.fitness))
    print("Avg fitness of gen: {}".format(avg_fit(gen_arr)))
    print("Fitness STD: {}".format(std_fit(gen_arr)))
    print("Total time of generation: {}".format(time.time() - timer))
    print("Total clock ticks (CPU)) of generation: {}\n".format(time.process_time() - ticks))


def birthday(gen_arr):
    for g in gen_arr:
        g.age += 1
    return gen_arr


def pso_alg(heu_type, init_values=range(32, 126)):
    gen_arr = init_population(init_values, pso=True)
    heu = heuristic_dictionary[heu_type]
    calc_fitness(gen_arr, heu)
    best_of_generation = min(gen_arr, key=lambda x: x.fitness)
    global_best = Genetic(best_of_generation.str, pso=True)
    global_best.fitness = heu.calc_personal_fitness(global_best)

    totaltimer = time.time()
    totalticks = time.process_time()

    for i in range(GA_MAXITER):
        gentimer = time.time()
        genticktimer = time.process_time()
        w = PSO_W_MAX * (1 - i / GA_MAXITER) + PSO_W_MIN * (i / GA_MAXITER)
        for g in gen_arr:
            g.pso_update(global_best, w, heu)
        best_of_generation = min(gen_arr, key=lambda x: x.fitness)
        if global_best.fitness > best_of_generation.fitness:
            global_best.str = best_of_generation.str
            global_best.fitness = heu.calc_personal_fitness(global_best)
        print_best(global_best, gen_arr, gentimer, genticktimer)
        if global_best.fitness == 0:
            break

    print("Total time : {}\nTotal clock ticks : {}\nTotal iter:{}".format(time.time() - totaltimer, time.process_time() - totalticks, i+1))


def gen_alg(cross_type, heu_type, select_type, init_values=range(32, 126)):
    gen_arr, buffer = init_population(init_values)
    heu = heuristic_dictionary[heu_type]
    cross = crossover_dictionary[cross_type]
    select = selection_dictionary[select_type]

    totaltimer = time.time()
    totalticks = time.process_time()

    for i in range(GA_MAXITER):
        gentimer = time.time()
        genticktimer = time.process_time()

        calc_fitness(gen_arr, heu)
        gen_arr = sort_by_fitness(gen_arr)
        print_best(gen_arr[0], gen_arr, gentimer, genticktimer)
        if gen_arr[0].fitness == 0:
            break

        gen_arr = birthday(gen_arr)
        # mate and swap between buffer and gen_arr
        buffer, gen_arr = mate(gen_arr, buffer, cross, select)

    print("Total time : {}\nTotal clock ticks : {}\nTotal iter:{}".format(time.time() - totaltimer, time.process_time() - totalticks, i+1))


pso_dictionary = {0: True, 1: False}


def main():
    do_pso = pso_dictionary[int(input("PSO - 0 / Normal - 1\n"))]
    heu = int(input("Choose heuristic:\n0 - Letter Distance\n1 - Hit Bonus\n"))
    if do_pso:
        pso_alg(heu)
        return
    cross = int(input("Choose crossover Type:\n0 - Uniform crossover\n1 - Single point cross over\n2 - Two point cross over\n"))
    select = int(input("Choose selection:\n0 - RWS\n1 - SUS\n2 - TOURNAMENT\n3 - REGULAR\n"))

    gen_alg(cross, heu, select)


if __name__ == "__main__":
    main()
