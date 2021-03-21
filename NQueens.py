import GeneticAlgorithm as GA
import numpy as np
import random
import time

Q_MAXITER = 100
NQ_POPSIZE = 200
N = 20


# PMX crossover from moodle
class PMXCrossover:
    def crossover(self, perm1, perm2, tar_len):
        iter = random.randint(1, N/2)
        child = [i for i in perm1]

        # do PMX crossover a random number of times
        for it in range(iter):
            index = random.randint(0, N-1)
            val1 = perm1[index]
            val2 = perm2[index]
            for i in range(N):
                if child[i] == val1:
                    child[i] = val2
                elif child[i] == val2:
                    child[i] = val1
        return child


# OX crossover from moodle
class OXCrossover:
    def crossover(self, perm1, perm2, tar_len):
        child = [i for i in perm1]
        # val1 will store half of the values randomly
        val1 = random.sample(range(1, N+1), N // 2)
        # val2 will store the remaining values
        val2 = [i for i in perm2 if i not in val1]
        v2 = 0
        for i in range(len(child)):
            if child[i] not in val1:
                child[i] = val2[v2]
                v2 += 1
        return child


crossover_dictionary = {'0': OXCrossover(), '1': PMXCrossover()}


class ConflictsTotal:
    def calc_fitness(self, boards):
        for b in boards:
            self.calc_personal_fitness(b)

    def calc_personal_fitness(self, board):
        total_fit = 0
        for i in range(1, N+1):
            for j in range(i+1, N+1):
                if board.str[i-1] == board.str[j-1] or\
                        board.str[i-1] == board.str[j-1] - (j - i) or\
                        board.str[i-1] == board.str[j-1] + (j - i):
                    total_fit += 1
        board.fitness = total_fit


heuristic_dictionary = {'0': ConflictsTotal()}


class SwapMutation:
    def mutate(self, gen, tar_len, init_values):
        index = random.sample(init_values, 2)
        temp = gen.str[index[0]-1]
        gen.str[index[0]-1] = gen.str[index[1]-1]
        gen.str[index[1]-1] = temp


class ScrambleMutation:
    def mutate(self, gen, tar_len, init_values):
        start = random.randint(0, tar_len)
        end = random.randint(start, tar_len)
        gen.str[start:end] = np.random.permutation(gen.str[start:end])


mut_dictionary = {'0': SwapMutation(), '1': ScrambleMutation()}


def init_nqueens():
    pop, buffer = [], []
    for i in range(NQ_POPSIZE):
        ran_prem = np.random.permutation(range(1, N+1))
        pop.append(GA.Genetic(ran_prem))
        buffer.append(GA.Genetic(ran_prem))
    return pop, buffer


def main_nqueens(mut, cross, select):
    boards, buffer = init_nqueens()
    # q_mut = mut_dictionary[input("Swap Mutation - 0 / Scramble Mutation - 1\n")]
    # q_cross = crossover_dictionary[input("Order Crossover - 0 / Partially Matched Crossover - 1\n")]
    # q_select = GA.selection_dictionary[int(input("Choose selection:\n0 - RWS\n1 - SUS\n2 - TOURNAMENT\n3 - REGULAR\n"))]
    q_mut = mut_dictionary[str(mut)]
    q_cross = crossover_dictionary[str(cross)]
    q_select = GA.selection_dictionary[select]
    q_heu = heuristic_dictionary['0']

    totaltimer = time.time()

    for i in range(Q_MAXITER):
        gentimer = time.time()

        q_heu.calc_fitness(boards)
        boards = GA.sort_by_fitness(boards)
        GA.print_best(boards[0], boards, gentimer)
        if boards[0].fitness == 0:
            return i, time.time() - totaltimer
            #break

        boards = GA.birthday(boards)
        # mate and swap between buffer and boards
        buffer, boards = GA.mate(boards, buffer, q_cross, q_select, range(1, N+1), q_mut, tar_len=N)
    return -1, time.time() - totaltimer
    total_time = time.time() - total_timer
    print("Total time : {}\nTotal clock ticks : {}\nTotal iter:{}".format(total_time, total_time*cpu_freq()[0]*2**20, i+1))


if __name__ == '__main__':
    main_nqueens()
