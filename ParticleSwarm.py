import GeneticAlgorithm as GA


def main():
    heu = GA.heuristic_dictionary[input("Choose heuristic:\n0 - Letter Distance\n1 - Hit Bonus\n")]
    gen_arr, _ = GA.init_population(pso=True)
    for i in range(GA.GA_MAXITER):
        heu.calc_fitness(gen_arr)
        gen_arr = GA.sort_by_fitness(gen_arr)
        for g in gen_arr:
            g.pso_update(gen_arr[0], GA.PSO_W_MAX*(1 - i/GA.GA_MAXITER)+GA.PSO_W_MIN*(i/GA.GA_MAXITER))
        print("BEST:{} ({})".format(gen_arr[0].str, gen_arr[0].fitness))
        if gen_arr[0].fitness == 0:
            break
    print("Total iter:{}".format(i + 1))


if __name__ == "__main__":
    main()
