import numpy as np
import pytest
from test_predicates import get_me_dem_apples
from testing_utils import Apple

from rddl import Entity, Variable
from rddl.entities import Gripper, Location, ObjectEntity


class TestLocation(Location):

    def __init__(self):
        super().__init__(None)
        self._loc = np.random.randn(3)

    def _get_location(self):
        return self._loc


def test_basic_variable():
    v_loc1 = Variable(typ=Location, global_name="location1")
    v_loc2 = Variable(typ=Location, global_name="location2")

    # Check if variables are different
    assert v_loc1.id != v_loc2.id, "Variable should be different!"

    # Link variables and check if they have the same ID
    v_loc1.link_to(v_loc2)
    assert v_loc1.id == v_loc2.id, "Variables should be equal when linked"

    loc = TestLocation()

    # bind location to variable 1
    v_loc1.bind(loc)
    # check if both variables have the same location (because they are linked)
    assert np.allclose(v_loc1().location, v_loc2().location), "Variables should be equal when bound."

    # check if unbinding variable 1 causes variable 2 to be unbound
    v_loc1.unbind()
    assert v_loc2.value is None and not v_loc2.is_bound(), "Unbinding variable 1 should cause variable 2 to be unbound as well."

    # bind again
    v_loc2.bind(loc)
    assert np.allclose(v_loc1().location, v_loc2().location), "Variables should be equal when bound."

    # unlink variable 1; this should not affect variable 2
    v_loc1.unlink("location1_2")
    assert v_loc1.id == v_loc2.id and np.allclose(v_loc1().location, v_loc2().location), "Even after unlinking, the previously bound value should stay the same."
    v_loc1.unbind()
    assert v_loc2.is_bound(), "Unbinding variable 1 should not affect variable 2."

    with pytest.raises(AssertionError) as excinfo:
        v_loc2.link_to(v_loc1)
    assert "Cannot align variable" in str(excinfo.value), "Linking is forbidden if variable is already bound."


def test_typing():
    v1 = Variable(typ=Apple)
    assert v1.type is Apple

    v2 = Variable(typ=ObjectEntity)
    assert v2.type is ObjectEntity

    with pytest.raises(AssertionError) as excinfo:
        v1.link_to(v2)
    assert "Cannot align variable" in str(excinfo.value), "Cannot link variables of different types!"

    apple = Apple()
    v1.bind(apple)
    assert v1.is_bound(), "There should be no problem"
    v2.bind(apple)
    assert v2.is_bound(), "There should be no problem, Apple is subtype of ObjectEntity"

    v3 = Variable(typ=Gripper)
    with pytest.raises(AssertionError) as excinfo:
        v3.bind(apple)
    assert "type" in str(excinfo.value), "Cannot bind value of different type!"


if __name__ == "__main__":
    test_basic_variable()
    test_typing()
