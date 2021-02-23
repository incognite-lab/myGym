import pybullet as p
import pybullet_data as pd
import os

p.connect(p.DIRECT)
objects_dir_path = "/home/megi/git/myGym/myGym/envs/objects/"
list_all = os.listdir(objects_dir_path)
for folder in ['storage','household']: #list_all[]:
    models_dir_path = objects_dir_path+folder+'/obj/'
    list_files = os.listdir(models_dir_path)
    list_files.sort()
    for filename in list_files:

        if '.obj' in filename and 'vhacd' not in filename and 'knife' not in filename and 'bin' not in filename and 'box' not in filename  and 'bowl' not in filename and 'spatula' not in filename and 'fork' not in filename:
            filename = os.path.splitext(filename)[0]
            name_in = models_dir_path+filename+'.obj'
            name_out = models_dir_path+filename+'_vhacd.obj'
            name_log = models_dir_path+'log.txt'
            p.vhacd(name_in, name_out, name_log, alpha=0.04,resolution=5000000)
            print(name_out)