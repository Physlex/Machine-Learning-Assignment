"""
This module tests the simple model func. Basically a sanity test.
"""


import numpy as np
import logging as log

from src import model


def test_model_func():
    """
    Sanity test for model function, just to check it does what I think it does.
    """

    test_matrix_3x3 = np.arange(9.0).reshape(3, 3)
    test_weight_3x1 = np.arange(3.0).reshape(3, 1)
    assert np.shape(model(test_matrix_3x3, test_weight_3x1)) == (3, 1)
