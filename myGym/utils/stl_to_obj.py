import bpy
import sys
import os

override = {'selected_bases': list(bpy.context.scene.object_bases)}
bpy.ops.object.delete(override)

objects_dir_path = "/home/megi/git/myGym/myGym/envs/objects/toys/"
list_all = os.listdir(objects_dir_path)
for filename in list_all:
    if '.stl' in filename:
        filename = os.path.splitext(filename)[0]
        stl_in = objects_dir_path+filename+'.stl'
        obj_out = objects_dir_path+filename+'.obj'
        print(obj_out)

        bpy.ops.import_mesh.stl(filepath=stl_in, filter_glob="*.stl", global_scale=1, axis_forward='-Z', axis_up='Y')
        bpy.ops.export_scene.obj(filepath=obj_out, filter_glob="*.obj", global_scale=1, axis_forward='-Z', axis_up='Y')
        override = {'selected_bases': list(bpy.context.scene.object_bases)}
        bpy.ops.object.delete(override)
