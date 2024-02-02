import unittest

import numpy as np
from bitarray import bitarray

from FullSolution import FullSolution
from PS import PS
from SearchSpace import SearchSpace


class TestPS(unittest.TestCase):
    def test_init(self):

        init_tuple = (0, -1, 2)
        init_list = list(init_tuple)
        init_ndarray = np.array(init_tuple)
        init_gen = range(3)

        from_tuple = PS(init_tuple)

        def assertCorrect(initialiser, msg):
            new_ps = PS(initialiser)
            self.assertEqual(from_tuple, new_ps, msg)

        assertCorrect(init_list, 'Initialiser from list did not match')
        assertCorrect(init_ndarray, 'Initialiser from ndarray did not match')
        assertCorrect(init_gen, 'Initialiser from generator did not match')

    def test_empty(self):
        search_space = SearchSpace((2, 3, 4))
        empty = PS.empty(search_space)


        self.assertEquals(len(empty), len(search_space), 'Empty PS has invalid length')
        self.assertTrue(not empty.fixed_mask.any(), 'Empty PS has invalid mask')


    def test_from_and_toFS(self):
        fs = FullSolution((2, 3, 4))
        ps = PS.from_FS(fs)

        self.assertTrue(ps.fixed_mask.all(), 'PS.from_FS yielded a FS with an invalid mask')
        self.assertTrue(np.array_equal(ps.values, fs.values), 'PS.from_FS yielded a FS with mismatched values')


        self.assertTrue(ps.is_fully_fixed(), 'PS.from_FS(fs).is_fully_fixed results incorrect')


