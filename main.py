import unittest

from FullSolution import FullSolution
from PS import PS
from SearchSpace import SearchSpace

if __name__ == '__main__':
    ps: PS = PS([1, 2, 3, -1, -1])
    fs: FullSolution = FullSolution([1, 2, 3, 4, 5])


    print(f"The PS is {ps} and the fs is {fs}")
