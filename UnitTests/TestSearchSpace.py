import unittest

import numpy as np

from FullSolution import FullSolution
from SearchSpace import SearchSpace


class TestSearchSpace(unittest.TestCase):
    def test_initialisers(self):
        """ tests that all the intialisers work properly"""
        as_tuple = (2, 3, 4)
        as_list = list(as_tuple)
        as_ndarray = np.array(as_list)
        as_gen = range(2, 5)

        part_left = SearchSpace((2,))
        part_right = SearchSpace((3, 4))

        from_tuple = SearchSpace(as_tuple)
        from_list = SearchSpace(as_list)
        from_ndarray = SearchSpace(as_ndarray)
        from_gen = SearchSpace(as_gen)
        from_parts = SearchSpace.concatenate_search_spaces((part_left, part_right))

        self.assertEqual(from_tuple, from_list, 'constructor from list is invalid')
        self.assertEqual(from_tuple, from_ndarray, 'constructor from ndarray is invalid')
        self.assertEqual(from_tuple, from_gen, 'constructor from generator is invalid')
        self.assertEqual(from_tuple, from_parts, 'constructor from SS concatenation is invalid')

    def test_cumulative_offsets(self):
        search_space = SearchSpace((2, 3, 4))
        intended_offsets = np.array([0, 2, 5, 9])  # notice the inclusion of 0 and 9
        self.assertTrue(np.array_equal(search_space.precomputed_offsets, intended_offsets),
                        'invalid precomputed offsets')

    def test_random_full_solution(self):
        search_space = SearchSpace((2, 3, 4))

        def verify_fs(fs: FullSolution):
            values_are_non_negative: np.ndarray = fs.values >= 0
            values_are_within_cardinalities: np.ndarray = fs.values < search_space.cardinalities
            self.assertTrue(all(values_are_non_negative), 'a sampled FS contains negative values')
            self.assertTrue(all(values_are_within_cardinalities), 'a sampled FS contains values > cardinalities')

        for _ in range(100):
            verify_fs(FullSolution.random(search_space))
