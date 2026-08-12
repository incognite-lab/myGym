import os

import numpy as np
import pytest

os.environ["PYTHONPATH"] = os.path.join(os.getcwd(), "tests")
from testing_utils import Apple

None

from testing_utils import (EnvSimulator, Observer, create_approach_action,
                           create_gripper_and_apple, time_function)

from rddl import Entity, Operand


@pytest.mark.skip
def test_aa_cache(create_approach_action, create_gripper_and_apple):
    a = create_approach_action
    objects_for_approach = create_gripper_and_apple
    a.bind(objects_for_approach)

    # env = EnvSimulator([v.reference for v in objects_for_approach.values()])
    env = EnvSimulator([v for v in objects_for_approach.values()])
    observer = Observer()
    Entity.set_observation_getter(observer.get_observation)

    for i in range(10):
        obs = env.step([0])
        Operand.reset_cache()
        observer.set_observation(obs)

        d_times, d_results = [], []
        r_times, r_results = [], []
        for _ in range(10):
            dt, dv = time_function(a.decide)
            d_times.append(dt)
            d_results.append(dv)
            rt, rv = time_function(a.reward)
            r_times.append(rt)
            r_results.append(rv)

        print(f"Testing iteration {i}: decision {d_results[0]}, reward {r_results[0]}")
        print(f"Full runtimes: decision {d_times[0]}, reward {r_times[0]}")
        print(f"Cached runtimes (avg): decision {np.mean(d_times[1:])}, reward {np.mean(r_times[1:])}")
        assert np.allclose(d_results, d_results[0]), "d_result is not constant"
        assert np.allclose(r_results, r_results[0]), "r_result is not constant"
        assert d_times[0] > np.max(d_times[1:]), "First computation time is not longer than others"
        assert r_times[0] > np.max(r_times[1:]), "First computation time is not longer than others"

    print("Done!")


if __name__ == "__main__":
    test_aa_cache(create_approach_action(), create_gripper_and_apple())
