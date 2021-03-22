import GeneticAlgorithm as GA
import numpy as np

N = 25      # size of board


# recursive minimal conflicts algorithm
def MinimalConflictsAlgorithm(board, iter=0):
    # print current state of board
    print(board)
    change = False
    worst_queen = -1

    # checks if a better move was found, if not will check different queen with same number of conflicts
    while not change:
        max_conf = 0
        for i in range(worst_queen+1, N):
            q_i_conf = queen_conflict(board, i)
            if q_i_conf > max_conf:
                max_conf = q_i_conf
                worst_queen = i
        # this if will only be true for local minimum or optimal solution
        if max_conf == 0:
            for i in range(N):
                max_conf += queen_conflict(board, i)
            print("iterations: ", iter)
            print("conflicts: ", max_conf)
            return board
        board, change = move_worst_queen(board, worst_queen)
    return MinimalConflictsAlgorithm(board, iter+1)


# calculates each queens conflicts
def queen_conflict(board, q):
    conflicts = 0
    for i in range(N):

        # note: a diagonal conflict is considered to be worse than horizontal
        # conflict (achieved better results)
        if i != q:
            if board[i] == board[q]:
                conflicts += 1
            if board[i] == board[q] - (q - i) or \
                    board[i] == board[q] + (q - i):
                conflicts += 1.1
    return conflicts


# finds best position for worst queen if possible
def move_worst_queen(board, worst_queen):
    change = False
    best_board = np.copy(board)
    for i in range(N):
        board[worst_queen] = i
        if queen_conflict(board, worst_queen) < queen_conflict(best_board, worst_queen):
            best_board = np.copy(board)
            change = True
    return best_board, change


def main():
    MinimalConflictsAlgorithm(np.random.permutation(range(N)))


if __name__ == "__main__":
    main()
