import GeneticAlgorithm as GA
import numpy as np
import random
import time

N = 8


def MinimalConflictsAlgorithm(board):
    print(board)
    change = False
    worst_queen = 0
    while not change:
        max_conf = 0
        for i in range(worst_queen+1, N):
            q_i_conf = queen_conflict(board, i)
            if q_i_conf > max_conf:
                max_conf = q_i_conf
                worst_queen = i
        if max_conf == 0:
            return board
        board, change = move_worst_queen(board, worst_queen)
    return MinimalConflictsAlgorithm(board)


def queen_conflict(board, q):
    conflicts = 0
    for i in range(N):
        if i != q:
            if board[i] == board[q]:
                conflicts += 1
            if board[i] == board[q] - (q - i) or \
                    board[i] == board[q] + (q - i):
                conflicts += 1.1
    return conflicts


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
