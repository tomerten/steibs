# -*- coding: utf-8 -*-

"""Tests for steibs package."""

import ste
import STELib as stelib


def test_ste_random_ran3():
    assert (0.684440382 - stelib.ran3(1234)) < 1e-4


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
