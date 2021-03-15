import GeneticAlgorithm as GA
from typing import overload


def foo(x):
    print(x)


def foo(x,*args):
    print(x+)


foo(3,4)