import GeneticAlgorithm as GA
import numpy as np
import random
import time

Q_MAXITER = 10000
NQ_POPSIZE = GA.GA_POPSIZE
N = 15


# PMX crossover from moodle
class PMXCrossover:
    def crossover(self, perm1, perm2):
        iter = random.randint(1,N-1)
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


class OXCrossover:
    def crossover(self, perm1, perm2):
        child = [i for i in perm1]
        val1 = random.sample(range(0, N), N // 2)
        val2 = [i for i in perm2 if i not in val1]
        v1, v2 = 0, 0
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
        for i in range(N):
            for j in range(i+1,N):
                if board.str[i] == board.str[j] or board.str[i] == board.str[j] - (j - i) or board.str[i] == board.str[j] + (j - i):
                    total_fit += 1
        board.fitness = total_fit


heuristic_dictionary = {'0': ConflictsTotal()}


class SwapMutation:
    def mutate(self, gen):
        index = random.sample(range(0, N), 2)
        temp = gen.str[index[0]]
        gen.str[index[0]] = gen.str[index[1]]
        gen.str[index[1]] = temp


class ScrambleMutation:
    def mutate(self, gen):
        start = random.randint(0, N)
        end = random.randint(start, N)
        gen.str[start:end] = np.random.permutation(gen.str[start:end])


mut_dictionary = {'0': SwapMutation(), '1': ScrambleMutation()}


def init_nqueens():
    pop, buffer = [], []
    for i in range(NQ_POPSIZE):
        ran_prem = np.random.permutation(range(1, N+1))
        pop.append(GA.Genetic(ran_prem))
        buffer.append(GA.Genetic(ran_prem))
    return pop, buffer


def main_nqueens():
    boards, buffer = init_nqueens()
    q_mut = mut_dictionary[input("Swap Mutation - 0 / Scramble Mutation - 1\n")]
    q_cross = crossover_dictionary[input("Order Crossover - 0 / Partially Matched Crossover - 1\n")]
    q_select = GA.selection_dictionary[int(input("Choose selection:\n0 - RWS\n1 - SUS\n2 - TOURNAMENT\n3 - REGULAR\n"))]

    q_heu = heuristic_dictionary['0']

    totaltimer = time.time()
    totalticks = time.process_time()

    for i in range(Q_MAXITER):
        gentimer = time.time()
        genticktimer = time.process_time()

        q_heu.calc_fitness(boards)
        boards = GA.sort_by_fitness(boards)
        GA.print_best(boards[0], boards, gentimer, genticktimer)
        if boards[0].fitness == 0:
            break

        boards = GA.birthday(boards)
        # mate and swap between buffer and boards
        buffer, boards = GA.mate(boards, buffer, q_cross, q_select, q_mut)

    print("Total time : {}\nTotal clock ticks : {}\nTotal iter:{}".format(time.time() - totaltimer,
                                                                          time.process_time() - totalticks, i + 1))


if __name__ == '__main__':
    main_nqueens()
