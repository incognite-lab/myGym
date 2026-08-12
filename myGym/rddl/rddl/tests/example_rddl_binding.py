from testing_utils import TiagoGripper, Apple
from rddl.actions import Approach, Grasp, Drop, Move, Follow
from rddl.rddl_sampler import RDDLWorld
from rddl.entities import Gripper, ObjectEntity


if __name__ == "__main__":
    rddl_world = RDDLWorld(
        allowed_actions=[Approach, Grasp, Drop, Move, Follow],
        allowed_entities=[Apple, TiagoGripper]
    )

    gen = rddl_world.sample_generator(5)
    actions = []

    for _ in range(5):
        action = next(gen)
        actions.append(action)

        for v in rddl_world.get_created_variables():
            if v.is_bound():
                continue  # skip already bound variables
            if issubclass(v.type, Gripper):
                v.bind(TiagoGripper())
            elif issubclass(v.type, ObjectEntity):
                v.bind(Apple())
            else:
                raise ValueError(f"I cannot bind variable of type {v.type}, yet!")

        print(f"Current action: {action}\nCurrent reward: {action.compute_reward()}")

    print(f"Generated action sequence: {actions}")
