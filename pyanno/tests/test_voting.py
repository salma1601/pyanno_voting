import unittest
import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal)

from pyanno import voting
from pyanno.voting import MISSING_VALUE as MV
from pyanno.voting import PyannoValueError as PVE


class TestVoting(unittest.TestCase):
    def test_123(self):
        self.assertEqual(1+2, 3)

    def test_123_float(self):
        self.assertAlmostEqual(1.1+2.2, 3.3)
    def test_numpy_add(self):

        x = np.array([1,1])
        y = np.array([2,2])
        z = np.array([3,3])
        np.testing.assert_array_equal(x+y, z)
#       self.assertEqual(x+y,z) won't work because truth value ambig. 
#       self.assertEqual((x+y).all(),z.all()) would work

    def test_labels_count(self):
        annotations = [
            [1,  2, MV, MV],
            [MV, MV,  3,  3],
            [MV,  1,  3,  1],
            [MV, MV, MV, MV],
        ]
        nclasses = 5
        expected = [0, 3, 1, 3, 0]
        result = voting.labels_count(annotations, nclasses)
        self.assertEqual(result, expected)
    
    def test_labels_count_newmissingvalue(self):
        m = -99
        annotations = [
            [1,  2, m, m],
            [m, m,  3,  3],
            [m,  1,  3,  1],
            [m, m, m, m],
        ]
        nclasses = 5
        expected = [0, 3, 1, 3, 0]
        result = voting.labels_count(annotations, nclasses, m)
        self.assertEqual(result, expected)

    def test_majority_vote_newmissingvalue(self):
        m = -99
        annotations = [
            [1, 2, 2, m],
            [2, 2, 2, 2],
            [1, 1, 3, 3],
            [1, 3, 3, 2],
            [m, 2, 3, 1],
            [m, m, m, 3],
        ]
        expected = [2, 2, 1, 3, 1, 3]
        result = voting.majority_vote(annotations, m)
        self.assertEqual(expected, result)
    def test_majority_vote(self):
        annotations = [
            [1, 2, 2, MV],
            [2, 2, 2, 2],
            [1, 1, 3, 3],
            [1, 3, 3, 2],
            [MV, 2, 3, 1],
            [MV, MV, MV, 3],
        ]
        expected = [2, 2, 1, 3, 1, 3]
        result = voting.majority_vote(annotations)
        self.assertEqual(expected, result)

    def test_majority_vote_empty_item_newmissingvalue(self):
        MV = -99
        # Bug: majority vote with row of invalid annotations fails
        annotations = np.array(
            [[1, 2, 3],
             [MV, MV, MV],
             [1, 2, 2]]
        )
        expected = [1, MV, 2]
        result = voting.majority_vote(annotations, MV)
        self.assertEqual(expected, result)

    def test_majority_vote_empty_item(self):
        # Bug: majority vote with row of invalid annotations fails
        annotations = np.array(
            [[1, 2, 3],
             [MV, MV, MV],
             [1, 2, 2]]
        )
        expected = [1, MV, 2]
        result = voting.majority_vote(annotations)
        self.assertEqual(expected, result)

    def test_labels_frequency(self):
        annotations = [[1, 1, 2], [-1, 1, 2]]
        nclasses = 4
        expected_result = [0., 0.6, 0.4, 0.0]
        result = voting.labels_frequency(annotations, nclasses)
        #self.assertAlmostEqual(expected_result, list(result)) #AlmostEqual not working on np.arrays but lists
        assert_array_almost_equal(expected_result, result) #Probably better

    def test_labels_count_Error_MV(self):
        with self.assertRaises(PVE):
            annotations = np.array([[MV,MV,MV]])
            result = voting.labels_count(annotations, 3)
    
    def test_labels_count_Error_Empty(self):
        with self.assertRaises(PVE):
            result = voting.labels_count([], 3)


if __name__ == '__main__':
    unittest.main()
