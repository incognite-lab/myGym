import os

import numpy as np
import pytest

import rddl

# os.environ["PYTHONPATH"] = os.getcwd()
os.environ["PYTHONPATH"] = os.path.join(os.getcwd(), "tests")
from testing_utils import Apple

from rddl.operators import NotOp
from rddl.predicates import Near


# @pytest.fixture
def get_me_dem_apples():
    apple1 = Apple("apple1")
    apple2 = Apple("apple2")
    return apple1, apple2


def test_predicate_near(get_me_dem_apples):
    apple1, apple2 = get_me_dem_apples
    apples_dict = {"object_A": apple1, "object_B": apple2}

    near_bare = Near()

    near_reused = Near(object_A=near_bare.get_argument("object_A"), object_B=near_bare.get_argument("object_B"))

    # near_other = Near()

    near_bare.bind(apples_dict)

    near_bare.get_argument("object_A").global_rename("apple1")
    near_bare.get_argument("object_B").global_rename("apple2")

    assert near_bare.get_argument("object_A").name == "apple1", "Name should have been renamed to 'apple1'"
    assert near_bare.get_argument("object_B").name == near_reused.get_argument("object_B").name, "Names of near_bare and near_reused should be the same"

    with pytest.raises(AssertionError) as excinfo:
        near_reused.bind(apples_dict)
    assert "already bound" in str(excinfo.value), "Binding here should have risen an error since we are trying to rebind variables already bound above."

    assert near_bare() == near_reused()
    assert near_reused.object_A() is apple1
    assert near_reused.object_B() is apple2
    assert near_bare.object_A == near_reused.object_A
    assert near_bare.object_B == near_reused.object_B

    apple1.set_position(np.array([0, 0, 0]))
    apple2.set_position(np.array([0, 0, 0]))

    assert near_bare() and near_reused(), "Both near predicates should be true"

    apple1.set_position(np.array([0, 0, 1]))

    assert not (near_bare() or near_reused()), "Both near predicates should be false"


def test_compund_predicate_near(get_me_dem_apples):
    apple1, apple2 = get_me_dem_apples
    apples_dict = {"object_A": apple1, "object_B": apple2}
    apple1.set_position(np.array([0, 0, 0]))
    apple2.set_position(np.array([0, 0, 0]))

    near = Near()
    near.bind(apples_dict)
    assert near.decide()

    compund = NotOp(near)
    assert not compund.decide(), "Compound predicate 'Not(Near)' should be false"

    double_compund = NotOp(compund)
    assert double_compund.decide(), "Compound predicate 'Not(Not(Near))' should be true"


if __name__ == "__main__":
    test_predicate_near(get_me_dem_apples())
    test_compund_predicate_near(get_me_dem_apples())
