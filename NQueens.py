import GeneticAlgorithm as GA
import numpy as np

NQ_POPSIZE = 2048

class SwapMutation:


def init_nqueens():
    pop, buffer = [], []
    for i in range(NQ_POPSIZE):
        ran_prem = np.random.permutation(range(1, 9))
        pop.append(GA.Genetic(ran_prem))
        buffer.append(GA.Genetic(ran_prem))
    return pop, buffer



def main_nqueens():
    boards, buffer = init_nqueens()
    q_heu = heuristic_dictionary[heu_type]
    q_cross = crossover_dictionary[cross_type]
    q_select = selection_dictionary[select_type]

    totaltimer = time.time()
    totalticks = time.process_time()

    for i in range(GA_MAXITER):
        gentimer = time.time()
        genticktimer = time.process_time()

        q_heu.calc_fitness(gen_arr)
        gen_arr = sort_by_fitness(gen_arr)
        print_best(gen_arr[0], gen_arr, gentimer, genticktimer)
        if gen_arr[0].fitness == 0:
            break


        gen_arr = birthday(gen_arr)
        # mate and swap between buffer and gen_arr
        buffer, gen_arr = mate(gen_arr, buffer, q_cross, q_select)

    print("Total time : {}\nTotal clock ticks : {}\nTotal iter:{}".format(time.time() - totaltimer, time.process_time() - totalticks, i+1))


if __name__ == '__main__':
    main_nqueens()