import os

import numpy as np
import pytest

import rddl

# os.environ["PYTHONPATH"] = os.getcwd()
os.environ["PYTHONPATH"] = os.path.join(os.getcwd(), "tests")
from testing_utils import Apple

from rddl.core import Operand

None
from rddl.operators import NotOp, ParallelAndOp, SequentialOp
from rddl.predicates import Near


# @pytest.fixture
def get_me_dem_apples():
    apple1 = Apple("apple1")
    apple2 = Apple("apple2")
    return apple1, apple2


def get_me_dem_other_apples():
    apple1 = Apple("apple3")
    apple2 = Apple("apple4")
    return apple1, apple2


def test_predicate_near():
    Operand.set_cache_symbolic()
    near = Near()
    assert not near.decide(), "Predicates should be false by default"

    print("Setting near to false.")
    near.set_symbolic_value(True)
    assert near.decide(), "Near should be true after setting its symbolic value to True."


def test_predicate_near_multiple():
    Operand.set_cache_symbolic()
    near_bare = Near()
    near_reused = Near(object_A=near_bare.get_argument("object_A"), object_B=near_bare.get_argument("object_B"))
    near_other = Near()

    near_bare.set_symbolic_value(True)

    assert near_bare.decide(), "Near should be true after setting its symbolic value to True."
    assert near_reused.decide(), "Near should be true after setting its symbolic value to True."
    assert not near_other.decide(), "Near should be false after setting if it was not set before."


def test_compound_predicate_near():
    Operand.set_cache_symbolic()
    near = Near()

    compund = NotOp(near)
    assert compund.decide(), "Predicates should be false by default, applying Not(Near) should be true."

    double_compund = NotOp(compund)
    assert not double_compund.decide(), "Predicates should be false by default, applying Not(Not(Near)) should be false."

    near.set_symbolic_value(True)

    assert double_compund.decide(), "Compound predicate 'Not(Not(Near))' should be now true (Near symbolic set to True)"

    double_compund.set_symbolic_value(False)
    assert not double_compund.decide(), "Compound predicate 'Not(Not(Near))' false after setting it to False."
    assert not near.decide(), "Predicate 'Near' should be false after setting Not(Not(Near)) to False."


if __name__ == "__main__":
    test_predicate_near()
    test_predicate_near_multiple()
    test_compound_predicate_near()
