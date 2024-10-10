

import unittest
import gv_metamodel as gvtm
import numpy as np

class WindowAdapterTest(unittest.TestCase):

    def test_paths_exist(self):
        window_size = 5
        input_width = 55
        adapter = gvtm.WindowAdapter(window_size=window_size, input_width=input_width)

        self.assertRaises(AssertionError, adapter.get_window)

        adapter.add_input(np.zeros(input_width))

        expected = np.zeros((window_size, input_width))
        np.testing.assert_array_equal(expected, adapter.get_window())

        adapter.add_input(np.ones(input_width))

        expected = np.stack([np.zeros(input_width) for _ in range(window_size - 1)] + [np.ones(input_width)])
        np.testing.assert_array_equal(expected, adapter.get_window())

