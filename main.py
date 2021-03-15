import GeneticAlgorithm as GA


pso_dictionary = {0: True, 1: False}


def main():
    do_pso = pso_dictionary[int(input("PSO - 0 / Normal - 1\n"))]
    heu = int(input("Choose heuristic:\n0 - Letter Distance\n1 - Hit Bonus\n"))
    if do_pso:
        GA.gen_alg_PSO(heu, range(32,126))
        return
    cross = int(input("Choose crossover Type:\n0 - Uniform crossover\n1 - Single point cross over\n2 - Two point cross over\n"))
    select = int(input("Choose selection:\n0 - RWS\n1 - SUS\n2 - TOURNAMENT\n3 - REGULAR\n"))

    GA.gen_alg(cross, heu, select, range(32,126))


if __name__ == "__main__":
    main()
