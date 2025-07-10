#from myGym.envs.igibson_predicates import *
from myGym.envs import env_object
from myGym.envs.test_volume_class import VolumeMesh
import pybullet as p
import open3d as o3d
import numpy as np


class IsReachable():
    # Just checks if object is in predefined theoretically reachable envelope, but not actual reachability!
    def set_value(self, gripper, obj=None):
        robot_base = list(gripper.position)
        reachable_range = gripper.reachable_range_from_base
        actual_range = []
        for n in range(3):
            actual_range.append([robot_base[n] + reachable_range[n][0], robot_base[n] + reachable_range[n][1]])
        return actual_range

    def get_value(self, gripper, obj):
        robot_base = list(gripper.position)
        reachable_range = gripper.reachable_range_from_base
        obj_position = list(obj.location)
        for id, p in enumerate(obj_position):
            actual_range = [robot_base[id] + reachable_range[id][0], robot_base[id] + reachable_range[id][1]]
            if p < actual_range[0] or p >  actual_range[1]:
                return False
        return True 

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

  
def get_range_intersection(rangeA, rangeB) -> list:
    intersection = []
    for i in range(3):
        min_val = max(rangeA[i][0], rangeB[i][0])
        max_val = min(rangeA[i][1], rangeB[i][1])
        # Check if there is a valid intersection
        if min_val <= max_val:
            intersection.append([min_val, max_val])
        else:
            # No overlap for this axis
            intersection.append(None)
    return intersection
    
    
def get_scale_from_urdf(pth):
    with open(pth) as f:
        lines = f.readlines()
    if len([x for x in lines if "scale" in x]) > 0:
        scale = float([x for x in lines if "scale" in x][0].split("scale=\"")[1].split(" ")[0])
    else:
        scale = 1
    return scale



if __name__ == '__main__':
    import gymnasium as gym
    import importlib.resources as pkg_resources
    import argparse
    import os, commentjson
    from myGym import envs

    def configure_env(arg_dict, model_logdir=None, for_train=True):
        env_arguments = {"render_on": True, "visualize": arg_dict["visualize"], "workspace": arg_dict["workspace"], "framework":"stable_baselines",
                        "robot": arg_dict["robot"], "robot_init_joint_poses": arg_dict["robot_init"],
                        "robot_action": arg_dict["robot_action"],"max_velocity": arg_dict["max_velocity"],
                        "max_force": arg_dict["max_force"],
                        "action_repeat": arg_dict["action_repeat"], "rddl": arg_dict["rddl"],
                        "observation":arg_dict["observation"], "distractors":arg_dict["distractors"],
                        "num_networks":arg_dict.get("num_networks", 1), "network_switcher":arg_dict.get("network_switcher", "gt"),
                        "active_cameras": arg_dict["camera"], "color_dict":arg_dict.get("color_dict", {}),
                        "max_steps": arg_dict["max_episode_steps"], "visgym":arg_dict["visgym"],
                        "logdir": arg_dict["logdir"], "vae_path": arg_dict["vae_path"],
                        "yolact_path": arg_dict["yolact_path"], "yolact_config": arg_dict["yolact_config"],
                        "natural_language": bool(arg_dict["natural_language"]),
                        "training": bool(for_train)
                        }
        env_arguments["gui_on"] = arg_dict["gui"]
        env = gym.make(arg_dict["env_name"], **env_arguments)
        return env

    def get_parser():
            parser = argparse.ArgumentParser()
            #Envinronment
            parser.add_argument("-cfg", "--config", default="configs/train_prag.json", help="Can be passed instead of all arguments")
            parser.add_argument("-n", "--env_name", type=str, help="The name of environment")
            parser.add_argument("-ws", "--workspace", type=str, help="The name of workspace")
            parser.add_argument("-p", "--engine", type=str,  help="Name of the simulation engine you want to use")
            parser.add_argument("-d", "--render", type=str,  help="Type of rendering: opengl, opencv")
            parser.add_argument("-c", "--camera", type=int, help="The number of camera used to render and record")
            parser.add_argument("-vi", "--visualize", type=int,  help="Whether visualize camera render and vision in/out or not: 1 or 0")
            parser.add_argument("-vg", "--visgym", type=int,  help="Whether visualize gym background: 1 or 0")
            parser.add_argument("-g", "--gui", type=int, help="Wether the GUI of the simulation should be used or not: 1 or 0")
            #Robot
            parser.add_argument("-b", "--robot", type=str, help="Robot to train: kuka, panda, jaco ...")
            parser.add_argument("-bi", "--robot_init", nargs="*", type=float, help="Initial robot's end-effector position")
            parser.add_argument("-ba", "--robot_action", type=str, help="Robot's action control: step - end-effector relative position, absolute - end-effector absolute position, joints - joints' coordinates")
            #Task
            parser.add_argument("-tt", "--task_type", type=str,  help="Type of task to learn: reach, push, throw, pick_and_place")
            parser.add_argument("-ns", "--num_subgoals", type=int, help="Number of subgoals in task")
            parser.add_argument("-to", "--task_objects", nargs="*", type=str, help="Object (for reach) or a pair of objects (for other tasks) to manipulate with")
            parser.add_argument("-u", "--used_objects", nargs="*", type=str, help="List of extra objects to randomly appear in the scene")
            parser.add_argument("-oa", "--object_sampling_area", nargs="*", type=float, help="Area in the scene where objects can appear")
            #Distractors
            parser.add_argument("-di", "--distractors", type=str, help="Object (for reach) to evade")
            parser.add_argument("-dm", "--distractor_moveable", type=int, help="can distractor move (0/1)")
            parser.add_argument("-ds", "--distractor_constant_speed", type=int, help="is speed of distractor constant (0/1)")
            parser.add_argument("-dd", "--distractor_movement_dimensions", type=int, help="in how many directions can the distractor move (1/2/3)")
            parser.add_argument("-de", "--distractor_movement_endpoints", nargs="*", type=float, help="2 coordinates (starting point and ending point)")
            parser.add_argument("-no", "--observed_links_num", type=int, help="number of robot links in observation space")
            #Reward
            parser.add_argument("-rt", "--reward_type", type=str, help="Type of reward: gt(ground truth), 3dvs(3D vision supervised), 2dvu(2D vision unsupervised), 6dvs(6D vision supervised)")
            parser.add_argument("-re", "--reward", type=str,  help="Defines how to compute the reward")
            parser.add_argument("-dt", "--distance_type", type=str, help="Type of distance metrics: euclidean, manhattan")
            #Train
            parser.add_argument("-w", "--train_framework", type=str,  help="Name of the training framework you want to use: {tensorflow, pytorch}")
            parser.add_argument("-a", "--algo", type=str,  help="The learning algorithm to be used (ppo2 or her)")
            parser.add_argument("-s", "--steps", type=int, help="The number of steps to train")
            parser.add_argument("-ms", "--max_episode_steps", type=int,  help="The maximum number of steps per episode")
            parser.add_argument("-ma", "--algo_steps", type=int,  help="The number of steps per for algo training (PPO2,A2C)")
            #Evaluation
            parser.add_argument("-ef", "--eval_freq", type=int,  help="Evaluate the agent every eval_freq steps")
            parser.add_argument("-e", "--eval_episodes", type=int,  help="Number of episodes to evaluate performance of the robot")
            #Saving and Logging
            parser.add_argument("-l", "--logdir", type=str,  help="Where to save results of training and trained models")
            parser.add_argument("-r", "--record", type=int, help="1: make a gif of model perfomance, 2: make a video of model performance, 0: don't record")
            #Mujoco
            parser.add_argument("-i", "--multiprocessing", type=int,  help="True: multiprocessing on (specify also the number of vectorized environemnts), False: multiprocessing off")
            #Paths
            parser.add_argument("-m", "--model_path", type=str, help="Path to the the trained model to test")
            parser.add_argument("-vp", "--vae_path", type=str, help="Path to a trained VAE in 2dvu reward type")
            parser.add_argument("-yp", "--yolact_path", type=str, help="Path to a trained Yolact in 3dvu reward type")
            parser.add_argument("-yc", "--yolact_config", type=str, help="Path to saved config obj or name of an existing one in the data/Config script (e.g. 'yolact_base_config') or None for autodetection")
            parser.add_argument('-ptm', "--pretrained_model", type=str, help="Path to a model that you want to continue training")
            return parser

    def get_arguments(parser):
        args = parser.parse_args()
        if not os.path.exists(args.config):
            # try looking in the package main dir
            args.config = os.path.join(myGym.__path__[0], args.config)
            if not os.path.exists(args.config):
                raise ValueError("Could not find config file: {}".format(args.config))
        with open(args.config, "r") as f:
                arg_dict = commentjson.load(f)
        for key, value in vars(args).items():
            if value != None and key != "config":
                if key in ["robot_init", "object_sampling_area"]:
                    arg_dict[key] = [float(arg_dict[key][i]) for i in range(len(arg_dict[key]))]
                else:
                    arg_dict[key] = value
        return arg_dict

    parser = get_parser()
    arg_dict = get_arguments(parser)
    arg_dict["gui"] = 1
    table = env.env.env.static_scene_objects["table"]
    touching = Touching()
    on_top = OnTop()

    # load tuna model and check the scale
    pth1 = "./envs/objects/household/urdf/tuna_can.urdf"
    pos = env_object.EnvObject.get_random_object_position([-0.5, 0.5, 0.4, 0.6, 0.07, 0.07])
    obj1 = env_object.EnvObject(pth1, env, pos, [0, 0, 0, 1], pybullet_client=p, fixed=False)
    obj1_info = p.getVisualShapeData(obj1.get_uid())[0]
    obj1_scale = get_scale_from_urdf(pth1)
    objpth = obj1_info[4].decode("utf-8")

    # voxelize
    o3model = o3d.io.read_triangle_model(objpth)
    mesh = o3model.meshes[0].mesh
    mesh = mesh.scale(obj1_scale, center=mesh.get_center())
    vm = VolumeMesh(mesh)
    orig = vm.duplicate()
    orig.paint(np.array([0, 1, 0]))
    voxel_grid = vm.voxelgrid
    # visualize geometry
    #o3d.visualization.draw_geometries([voxel_grid, orig.voxelgrid])


    print("Tuna touching table:")
    print(touching.get_value(obj1, table))
    print("Tuna on top of table:")
    print(on_top.get_value(obj1, table))
    print("Table on top of tuna:")
    print(on_top.get_value(table, obj1))



