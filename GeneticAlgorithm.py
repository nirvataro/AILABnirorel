import sys
import random
import time
import numpy as np
from psutil import cpu_freq

############## constants ###############
GA_POPSIZE = 1000           # ga population size
GA_MAXITER = 2000   	    # maximum iterations
GA_ELITRATE = .1		    # elitism rate
GA_MUTATIONRATE = .25      # mutation rate
HIT_BONUS = 1
EXACT_BONUS = 10
PSO_C1 = 2
PSO_C2 = 2
PSO_W_MIN = .4
PSO_W_MAX = .9
PSO_LEARNING_RATE = .1
GA_TARGET = "Hello World!"
########################################


# class of genetic (GA_STRUCT in cpp example)
# compatible with PSO
class Genetic:
    def __init__(self, str_info, pso=False):
        self.str = str_info
        self.fitness = 0
        self.pso = pso
        self.age = 0

        # variables will be used only in pso mode
        if pso:
            self.p_best_str = str_info
            self.p_best_fitness = sys.maxsize
            self.velocity = [0 for i in range(len(GA_TARGET))]

    def pso_update(self, best, w, heu):
        self.velocity = [x * w for x in self.velocity]

        # pso calculation
        for l in range(len(GA_TARGET)):
            const_p = PSO_C1*random.random()
            const_g = PSO_C2*random.random()
            self.velocity[l] += const_p*(ord(self.p_best_str[l])-ord(self.str[l])) + const_g*(ord(best.str[l])-ord(self.str[l]))

        # result of pso iteration on gen
        res = [ord(self.str[i]) + round(PSO_LEARNING_RATE*self.velocity[i]) for i in range(len(GA_TARGET))]

        # bounding pso range to ascii chars
        for i in range(len(res)):
            if res[i] < 0:
                res[i] = 0
            if res[i] > 126:
                 res[i] = 126
        self.str = ''.join(chr(i) for i in res)
        self.fitness = heu.calc_personal_fitness(self, len(GA_TARGET))

        # update personal best if needed
        if self.fitness < self.p_best_fitness:
            self.p_best_fitness = self.fitness
            self.p_best_str = self.str


# generic fitness calculation - will calculate any type of gen given heuristic
def calc_fitness(gen_arr, heu, tar_len):
    for g in gen_arr:
        g.fitness = heu.calc_personal_fitness(g, tar_len)
        if g.pso and g.p_best_fitness > g.fitness:
            g.p_best_fitness = g.fitness
            g.p_best_str = g.str


########## Part 1 - EX4: classes for each type of fitness calculating method ##########
# will calculate a personal fitness for a given gen and assign that gen its fitness
class HitBonus:
    def calc_personal_fitness(self, gen, tar_len):
        # gives a basic fitness and from there gives bonus using "Bul Pgiya" method
        LDcalc = LetterDistance()
        temp_fit = LDcalc.calc_personal_fitness(gen, tar_len)
        # max possible bonus for a given string
        temp_fit += ((EXACT_BONUS + HIT_BONUS) * tar_len)

        # "Bul Pgiya" bonus calculator
        for i in range(tar_len):
            if gen.str[i] == GA_TARGET[i]:
                temp_fit -= EXACT_BONUS
            if gen.str[i] in GA_TARGET:
                temp_fit -= HIT_BONUS
        return temp_fit


# generic letter distance heuristic class
class LetterDistance:
    def calc_personal_fitness(self, gen, tar_len):
        fitness = 0
        gen.str = "".join([i for i in gen.str])
        for i in range(tar_len):
            fitness += abs(ord(gen.str[i]) - ord(GA_TARGET[i]))
        return fitness


# this dictionary will use to create heuristic class for calculating fitness, can be expanded in feature
heuristic_dictionary = {0: LetterDistance(), 1: HitBonus()}
#######################################################################################

################## Part 1 - EX3: classes for each type of crossover ##################
# one point crossover class
class OneCross:
    def crossover(self, str1, str2, tar_len):
        pos = random.randint(0, tar_len)
        return str1[:pos] + str2[pos:]


# two point crossover class
class TwoCross:
    def crossover(self, str1, str2, tar_len):
        pos1 = random.randint(0, tar_len)
        pos2 = random.randint(pos1, tar_len)
        return str1[:pos1] + str2[pos1:pos2] + str1[pos2:]


# Uniform crossover class
class UniCross:
    def crossover(self, str1, str2, tar_len):
        mutation = [random.choice([str1[i], str2[i]]) for i in range(tar_len)]
        return mutation


# this dictionary will use to determine what crossover type to use, can be expanded in feature
crossover_dictionary = {0: UniCross(), 1: OneCross(), 2: TwoCross()}
######################################################################################


################## Part 2 - EX1: selection + scaling implementation ##################
# for scaling the problem and creating a max instead min (for RWS, SUS) we used 1/sqrt(fitness scaling)
def scale(gen_arr):
    scaled_fit = [None for i in range(len(gen_arr))]
    for i in range(len(gen_arr)):
        scaled_fit[i] = 1/gen_arr[i].fitness**0.5
    return sum(scaled_fit), scaled_fit


# implementation of RWS selection type (uses scaling function above)
class RWS:
    def selection(self, gen_arr, parent=2):
        # sel will store selected parents to mate
        sel = [None for i in range(parent)]
        total_fitness, scaled_fit = scale(gen_arr)
        for i in range(parent):
            # randomize a number between 0-sum of all fitness, find where that gen is and use as parent
            ran_selection = random.uniform(0, total_fitness)
            current, j = 0, 0
            while current < ran_selection:
                current += gen_arr[j].fitness
                j += 1
            sel[i] = gen_arr[j]
        return sel


# implementation of SUS selection type (uses scaling function above)
class SUS:
    def selection(self, gen_arr, parent=2):
        sel = [None for i in range(parent)]
        total_fitness, scaled_fit = scale(gen_arr)
        # randomize only once as written in algorithm
        # similar to implementation of RWS
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


# implementation of tournament selection type
class TOURNAMENT:
    def selection(self, gen_arr, k_tour_size, parent=2):
        sel = [None for i in range(parent)]
        for i in range(parent):
            # selects k random parents and uses best of them fot mating
            tournament = random.choices(gen_arr, k=k_tour_size)
            sel[i] = min(tournament, key=lambda x: x.fitness)
        return sel


# regular selection as seen in cpp code
class REGULAR:
    def selection(self, gen_arr, k):
        return [gen_arr[random.randint(0, int(GA_POPSIZE/2))] for i in range(k)]


selection_dictionary = {0: RWS(), 1: SUS(), 2: TOURNAMENT(), 3: REGULAR()}
######################################################################################


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


# Part 2 - Ex.2 creating array of possible parents by age
def ageing(gen_arr, min_age):
    can_mate = []
    for g in gen_arr:
        if g.age >= min_age:
            can_mate.append(g)
    return can_mate


# age updater for every iteration
def birthday(gen_arr):
    for g in gen_arr:
        g.age += 1
    return gen_arr


# randomly changes one of the characters
class RandomMutate:
    def mutate(self, gen, tar_len, choices):
        pos = random.randint(0, tar_len-1)
        delta = random.choice(choices)
        new_str = list(gen.str)
        new_str[pos] = chr(delta)
        gen.str = new_str


# generic mating function
# supports elitism, aging, selection types, crossovers, possible string values, different target lengths
def mate(gen_arr, buffer, crossover_type, selection_type, init_values, mut_method=RandomMutate(), min_age=0, tar_len=len(GA_TARGET)):
    esize = int(GA_POPSIZE * GA_ELITRATE)       # number of elitism moving to next generation
    buffer = elitism(gen_arr, buffer, esize)    # filling buffer with best of this generation
    can_mate = ageing(gen_arr, min_age)         # generate possible parents

    # mating parents
    if len(can_mate) > 0:
        for i in range(esize, GA_POPSIZE):
            s = selection_type.selection(gen_arr, 2)
            mut = crossover_type.crossover(s[0].str, s[1].str, tar_len)
            buffer[i] = Genetic(mut)

            # in GA_MUTATIONRATE chance new child will mutate
            if np.random.choice([True, False], p=[GA_MUTATIONRATE, 1-GA_MUTATIONRATE]):
                mut_method.mutate(buffer[i], tar_len, init_values)
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
def print_best(best, gen_arr, timer):
    print("Best: {} ({}).".format(best.str, best.fitness))
    print("Avg fitness of gen: {}".format(avg_fit(gen_arr)))
    print("Fitness STD: {}".format(std_fit(gen_arr)))
    iter_time = time.time() - timer
    print("Total time of generation: {}".format(iter_time))
    print("Total clock ticks (CPU)) of generation: {}\n".format(iter_time*cpu_freq()[0]*(2**20)))


def pso_alg(heu_type, init_values=range(32, 126), tar_len=len(GA_TARGET)):
    gen_arr = init_population(init_values, pso=True)
    heu = heuristic_dictionary[heu_type]
    calc_fitness(gen_arr, heu, tar_len)
    best_of_generation = min(gen_arr, key=lambda x: x.fitness)
    global_best = Genetic(best_of_generation.str, pso=True)
    global_best.fitness = heu.calc_personal_fitness(global_best, tar_len)

    total_timer = time.time()

    for i in range(GA_MAXITER):
        gen_timer = time.time()
        w = PSO_W_MAX * (1 - i / GA_MAXITER) + PSO_W_MIN * (i / GA_MAXITER)
        for g in gen_arr:
            g.pso_update(global_best, w, heu)
        best_of_generation = min(gen_arr, key=lambda x: x.fitness)
        if global_best.fitness > best_of_generation.fitness:
            global_best.str = best_of_generation.str
            global_best.fitness = heu.calc_personal_fitness(global_best, tar_len)
        print_best(global_best, gen_arr, gen_timer)
        if global_best.fitness == 0:
            break
    total_time = time.time() - total_timer
    print("Total time : {}\nTotal clock ticks : {}\nTotal iter:{}".format(total_time, total_time*cpu_freq()[0]*2**20, i+1))


def gen_alg(cross_type, heu_type, select_type, tar_len=len(GA_TARGET), init_values=range(32, 126)):
    gen_arr, buffer = init_population(init_values)
    heu = heuristic_dictionary[heu_type]
    cross = crossover_dictionary[cross_type]
    select = selection_dictionary[select_type]
    total_timer = time.time()

    for i in range(GA_MAXITER):
        gen_timer = time.time()

        calc_fitness(gen_arr, heu, tar_len)
        gen_arr = sort_by_fitness(gen_arr)
        print_best(gen_arr[0], gen_arr, gen_timer)
        if gen_arr[0].fitness == 0:
            break

        gen_arr = birthday(gen_arr)
        # mate and swap between buffer and gen_arr
        buffer, gen_arr = mate(gen_arr, buffer, cross, select, init_values)

    total_time = time.time() - total_timer
    print("Total time : {}\nTotal clock ticks : {}\nTotal iter:{}".format(total_time, total_time*cpu_freq()[0]*2**20, i+1))


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
