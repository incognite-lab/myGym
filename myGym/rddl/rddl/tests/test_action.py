import testing_utils
from rddl.actions import Approach, ApproachReward


def test_reward_action():
    aa = Approach()
    aa._reward.get_observations()[0]()
    aa.compute_reward()


if __name__ == "__main__":
    test_reward_action()
