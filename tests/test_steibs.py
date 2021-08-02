# -*- coding: utf-8 -*-

"""Tests for steibs package."""

import numpy as np
import ste
import STELib as stelib


def test_ste_random_ran3():
    assert (0.684440382 - stelib.ran3(1234)) < 1e-4


def test_ste_random_BiGaussian4D():
    expected = np.array([-1.56048843e-05, -1.56227465e-05, -1.02344229e-05, -9.64431365e-06])
    actual = np.array(stelib.BiGaussian4D(1.0, 1e-9, 2, 1e-10, 1234))
    assert np.allclose(expected, actual)


# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (otherwise all tests are normally run with pytest)
# Make sure that you run this code with the project directory as CWD, and
# that the source directory is on the path
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_ste_random_ran3()

    print("__main__ running", the_test_you_want_to_debug)
    the_test_you_want_to_debug()
    print("-*# finished #*-")

# eof
