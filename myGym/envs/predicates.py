#from myGym.envs.igibson_predicates import *
from myGym.envs import env_object
from myGym.envs.test_volume_class import VolumeMesh
import pybullet as p
import open3d as o3d
import numpy as np

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


class IsReachable():
    def set_value(self, obj, robot):
        raise NotImplementedError()

    def get_value(self, obj, robot):
        pass

    
    
    
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
    from myGym.train import configure_env, get_parser, get_arguments
    from myGym import envs
    parser = get_parser()
    arg_dict = get_arguments(parser)
    arg_dict["gui"] = 1
    env = configure_env(arg_dict)
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



