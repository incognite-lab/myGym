#from myGym.envs.igibson_predicates import *
from myGym.envs import env_object
import pybullet as p


class Touching():
    def set_value(self, obj1, obj2):
        raise NotImplementedError()

    def get_value(self, obj1, obj2):
        overlap_objs = p.getOverlappingObjects(obj1.get_bounding_box()[0], obj1.get_bounding_box()[4])
        overlapping = list(o[0] for o in overlap_objs)
        return obj2.uid in overlapping


class OnTop():
    def set_value(self, obj1, obj2):
        raise NotImplementedError()

    def get_value(self, obj1, obj2):
        overlap_objs = p.getOverlappingObjects(obj1.get_bounding_box()[0], obj1.get_bounding_box()[4])
        overlapping = list(o[0] for o in overlap_objs)
        base1 = obj1.get_bounding_box()[-1][-1]
        base2 = obj2.get_bounding_box()[-1][-1]
        return obj2.uid in overlapping and base1 > base2

    





if __name__ == '__main__':
    import gym
    from myGym.train import configure_env, get_parser, get_arguments
    from myGym import envs
    parser = get_parser()
    arg_dict = get_arguments(parser)
    arg_dict["gui"] = 1
    env = configure_env(arg_dict)
    #env.reset()
    table = env.env.env.static_scene_objects["table"]
    touching = Touching()
    on_top = OnTop()
    pos = env_object.EnvObject.get_random_object_position([-0.5, 0.5, 0.4, 0.6, 0.045, 0.045])
    pan1 = env_object.EnvObject("myGym/envs/objects/household/urdf/pan.urdf", pos, [0, 0, 0, 1], pybullet_client=p, fixed=False)
    pan1.set_color([0,1,0,1])
    print("Green pan touching table:")
    print(touching.get_value(pan1, table))
    print("Green pan on top of table:")
    print(on_top.get_value(pan1, table))
    print("Table on top of green pan:")
    print(on_top.get_value(table, pan1))

    pos = env_object.EnvObject.get_random_object_position([-0.5, 0.5, 0.4, 0.6, 0.5, 0.55])
    pan2 = env_object.EnvObject("myGym/envs/objects/household/urdf/pan.urdf", pos, [0, 0, 0, 1], pybullet_client=p, fixed=False)
    pan2.set_color([1,0,0,1])
    print("Red pan touching table:")
    print(touching.get_value(pan2, table))
    print("")


