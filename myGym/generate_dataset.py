## script to generate train/test sets from the simulator in COCO or DOPE format. Used for vision training.

import gym
from myGym import envs
from matplotlib.pyplot import imshow, show
import cv2
import numpy as np
import os
import glob
import json
import commentjson
import sys
import random
from pycocotools.cocostuffhelper import segmentationToCocoMask, segmentationToCocoResult
from pycocotools import mask
import pybullet as p
from bbox import BBox3D
from myGym.envs.wrappers import RandomizedEnvWrapper
import pkg_resources

# config, specify here or pass as an input argument
CONFIG_DEFAULT = pkg_resources.resource_filename("myGym", 'configs/dataset_coco.json')

# helper functions:
def color_names_to_rgb():
    """
    Assign RGB colors to objects by name as specified in the training config file
    """
    with open(pkg_resources.resource_filename("myGym", 'utils/rgbcolors.json'), "r") as read_file:
        clr = json.load(read_file) #json file with suggested colors
        new_dict = {}
        for key, value in config['object_colors'].items():
            new_value = []
            for item in value:
                new_value.append(clr[item])
            new_dict[key] = new_value
        config['color_dict'] = new_dict

def _category_coco_format(): #COCO
    """
    Create list of dictionaries with category id-name pairs in MSCOCO format

    Returns:
        :return categories: (list) Categories in MSCOCO format
    """
    categories = []
    for value, key in config['used_class_names'].items():
        categories.append({"id": int(key), "name": str(value)})
    return categories

def _segmentationToPoly(mask, ):
    """
    Convert segmentation from RLE to polynoms ([[x1 y1 x2 x2 y2 ...]]). Code from https://github.com/facebookresearch/Detectron/issues/100#issuecomment-362882830.
    
    Parameters:
        :param mask: (array) Bitmap mask
        :return segmentationPoly: (list) Segmentation converted to polynoms
    """
    contours, _ = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentationPoly = []

    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:
            segmentationPoly.append(contour)
    return segmentationPoly

def create_coco_json(): #COCO
    """
    Create COCO json data structure

    Returns:
        :return data_train: (dict) Data structure for training data
        :return data_test: (dist) Data structure for testing data
    """
    data_train = dict(
            images=[# file_name, height, width, id
            ],
            type='instances',
            annotations=[# segmentation, area, iscrowd, image_id, bbox, category_id, id
            ],
            categories = _category_coco_format(),
    )
    data_test = dict(
            images=[# file_name, height, width, id
            ],
            type='instances',
            annotations=[# segmentation, area, iscrowd, image_id, bbox, category_id, id
            ],
            categories = _category_coco_format(),
    )
    return data_train, data_test

def create_3D_box(env_object,objdim): #DOPE
    objpos = env_object.get_position()
    objorient = env_object.get_orientation()
    #objdim = env_object.get_cuboid_dimensions()
    box= BBox3D(objpos[0],objpos[1],objpos[2],objdim[0],objdim[1],objdim[2],objorient[3],objorient[0],objorient[1],objorient[2])
    return box.p,box.center

class GeneratorCoco: #COCO
    """
    Generator class for COCO image dataset for YOLACT vision model training
    """
    def __init__(self):
        pass

    def get_env(self): #COCO
        """
        Create environment for COCO dataset generation according to dataset config file

        Returns:
            :return env: (object) Environment for dataset generation
        """
        env = RandomizedEnvWrapper(env=gym.make(config['env_name'],
            robot = config['robot'],
            render_on = True,
            gui_on = config['gui_on'],
            show_bounding_boxes_gui = config['show_bounding_boxes_gui'],
            changing_light_gui = config['changing_light_gui'],
            shadows_on = config['shadows_on'],
            color_dict = config['color_dict'],
            observation = config["observation"],
            used_objects = used_objects,
            task_objects = config["task_objects"],
            active_cameras = config['active_cameras'],
            camera_resolution = config['camera_resolution'],
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            dataset = True,
            ), 
            config_path = config['output_folder']+'/config_dataset.json')
        p.setGravity(0, 0, -9.81)
        return env

    def episode_zero(self):
        """
        Initial espisode set-up
        """
        self.id_unique = 0 #image_id*x for paralel dataset generation, otherwise 0

    def init_data(self): #COCO
        """
        Initialize data structures for COCO dataset annotations

        Returns:
            :return data_train: (dict) Data structure for training data
            :return data_test: (dist) Data structure for testing data
        """
        data_train, data_test = create_coco_json()
        return data_train, data_test

    def resume(self): #COCO
        """
        Resume COCO dataset generation

        Returns:
            :return data_train: (dict) Training data from preceding dataset generation in COCO data structure
            :return data_test: (dist) Testing data from preceding dataset generation in COCO data structure
            :return image_id: (int) ID of last generated image in preceding dataset generation
        """
        try:
            with open(os.path.join(config['output_test_folder'],'annotations.json'), 'r') as f:
                data_test = json.load(f)
        except:
            pass #happens when test JSON is empty, which can happen for small numbers
        try:
            with open(os.path.join(config['output_train_folder'],'annotations.json'), 'r') as f:
                data_train = json.load(f)
            # get the ID of last image
            img_ids = [img["id"] for img in data_train['images']]
            image_id = max(img_ids) +1 # +1 if last sample were test (thus not in here). it's safe to have holes in the ids, just need to monothically increase
            self.id_unique = len(data_test['annotations']) + len(data_train['annotations'])
            print("Resuming from image_id {} for episodes: {}".format(image_id, config['num_episodes']))
        except FileNotFoundError:
            image_id = 0
        return data_test, data_train, image_id

    def data_struct_image(self): #COCO
        """
        Assign name to COCO dataset image and train of test status

        Returns:
            :param data: (dict) Corresponding data dictionary
            :param name: (string) Name of image file for saving
        """
        data = data_test if isTestSample == True else data_train
        name = 'img_{}_cam{}.jpg'.format(image_id,camera_id)
        return data, name

    def store_image_info(self): #COCO
        """
        Append COCO dataset image info to corresponding data dict
        """
        data['images'].append(dict(
            file_name=name,
            height=im.shape[0],
            width=im.shape[1],
            id=image_id,))

    def get_append_annotations(self): #COCO
        """
        Make and append COCO annotations for each object in the scene
        """
        seg = segmentationToCocoMask(img_mask,object_uid)
        area = float(mask.area(seg))
        bbox = mask.toBbox(seg).flatten().tolist()
        #1 run length encoding RLE segmentation format
        seg['counts'] = str(seg['counts'], "utf-8") #utf-8 format in str
        #2 or poly segmentation format
        bitmap = mask.decode(seg)
        seg = _segmentationToPoly(bitmap)

        self.too_small_obj = False
        try:
            #notify and skip the object with too small visible representation
            assert(area > config['min_obj_area'])
            assert(len(seg)>0 and len(seg[0])>0)
        except:
            #make inverse map id->name  (just to pretty print)
            inv_map = dict(zip(config['used_class_names'].values(), config['used_class_names'].keys()))
            self.too_small_obj = inv_map[class_id]
        self.data_dict = dict(
                    id=self.id_unique,
                    image_id=image_id,
                    category_id=class_id,
                    segmentation=seg,
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
        if self.too_small_obj == False: #area ok
            data['annotations'].append(self.data_dict) #append annotations
            self.id_unique +=1
        else: #area too small to be realistically seen
            print('Too small object of class {} with area={} in img {}'.format(self.too_small_obj, self.data_dict['area'], name))

    def visualize(self): #COCO
        """
        Visualize mask and bounding box coordinates for COCO annotated object
        """
        mask = img_mask==object_uid
        mask = np.expand_dims(mask, axis=2)
        mask = 255*mask.astype('uint8')
        cv2.imshow('image',im)
        cv2.waitKey(1)
        if self.too_small_obj:
            cv2.imshow('Too small object', mask)
        else:
            cv2.imshow('Labeled object', mask)
        print("Object class: {}".format(class_name))
        cv2.waitKey(1000)
        #print(self.data_dict['bbox'])

    def write_json_end(self): #COCO
        """
        Write json file with COCO annotations to output directory
        """
        if config['make_dataset'] in ["new", "resume"]:
            print("Storing annotations.json at episode {} of {}.".format(episode, config['num_episodes']))
            for flag in ['test','train']:
                if flag == 'train':
                    folder = config['output_train_folder']
                    data = data_train
                else:
                    folder = config['output_test_folder']
                    data = data_test

                json_name = 'img_{}_cam{}.json'.format(image_id, camera_id)
                json_dict = {"images": data["images"], "type":'instances',"annotations": data["annotations"], "categories":_category_coco_format()}
                if len(data["images"]) > 0:
                    with open(os.path.join(folder,json_name), 'w') as f:
                         json.dump(json_dict, f, indent=4)

                # clear data and continue
                data["images"].clear()
                data["annotations"].clear()

class GeneratorDope: #DOPE
    def __init__(self):
        self.object_settings = {"exported_object_classes": [], "exported_objects": []}

    def get_env(self): #DOPE
        env = RandomizedEnvWrapper(env=gym.make(config['env_name'],
            robot = config['robot'],
            render_on = True,
            gui_on = config['gui_on'],
            show_bounding_boxes_gui = config['show_bounding_boxes_gui'],
            changing_light_gui = config['changing_light_gui'],
            shadows_on = config['shadows_on'],
            color_dict = config['color_dict'],
            object_sampling_area = config['object_sampling_area'],
            observation = config["observation"],
            used_objects = used_objects,
            task_objects = config["task_objects"],
            active_cameras = config['active_cameras'],
            camera_resolution = config['camera_resolution'],
            dataset = True,
            ), config_path = config['output_folder']+'/config_dataset.json')
        p.setGravity(0, 0, -9.81)
        return env

    def episode_zero(self):
        self.objdim = {}
        while len(self.objdim.keys()) < len(config['used_class_names'].keys()):
            env.reset(random_robot=config['random_arm_movement'], random_pos=False)
            observation = env.get_observation()
            env_objects = observation["objects"]
            e = list(env_objects.values())
            env_object_list = []
            for sublist in e:
                if isinstance(sublist, list):
                    for item in sublist:
                        env_object_list.append(item)
                else:
                    env_object_list.append(sublist)
            for obj in env_object_list:
                if obj.name not in self.objdim.keys():
                    if obj != env.robot:
                        self.objdim[obj.name] = obj.get_cuboid_dimensions()

    def init_data(self): #DOPE
        data_train = {"objects":[]}
        data_test = {"objects":[]}
        return data_train, data_test

    def resume(self): #DOPE
        try:
            files_test = [int(x.replace('.jpg','')) for x in os.listdir(config['output_test_folder']) if x.endswith(".jpg")]
            files_train = [int(x.replace('.jpg','')) for x in os.listdir(config['output_train_folder']) if x.endswith(".jpg")]
            image_id = max(max(files_test),max(files_train))
            print("Resuming from image_id {} for episodes: {}".format(image_id, config['num_episodes']))
            self.episode_zero()
        except FileNotFoundError:
            image_id = 0
        return self.init_data()[0],self.init_data()[1],image_id

    def data_struct_image(self): #DOPE
        data_train, data_test = self.init_data()
        data = data_test if isTestSample == True else data_train
        name = '{}.jpg'.format(image_id)
        return data, name

    def store_image_info(self): #DOPE
        #write dataset image info
        filename = str(image_id) + '.json'
        if config['make_dataset'] in ["new", "resume"]:
            print("Storing {} and {} at episode {} of {}.".format(filename, name, episode, config['num_episodes']))
            with open(os.path.join(path, filename), 'w') as f:
                json.dump(data, f, indent=4)

    def get_append_annotations(self): #DOPE
        cuboid_with_centroid = env_object.get_bounding_box()
        cuboid_centroid=cuboid_with_centroid[8]
        cuboid=cuboid_with_centroid[:8]
        seg = segmentationToCocoMask(img_mask, object_uid)
        seg['counts'] = str(seg['counts'], "utf-8") #utf-8 format in str
        bbox = mask.toBbox(seg).flatten().tolist()
        bounding_box = {'top_left': bbox[:2], 'bottom_right': [bbox[0]+bbox[2], bbox[1]+bbox[3]]}
        boxp,boxc = create_3D_box(env_object,self.objdim[class_name])
        box3D = []
        for x in range(boxp.shape[0]):
            box3D.append(tuple(boxp[x]))
        boxc = list(boxc)
        projected_cuboid_centroid = list(env.project_point_to_camera_image(cuboid_centroid, camera_id))
        projected_cuboid = [list(env.project_point_to_camera_image(point, camera_id)) for point in cuboid]
        projected_3DBB_centroid = list(env.project_point_to_camera_image(boxc, camera_id))
        projected_3DBB = [list(env.project_point_to_camera_image(point, camera_id)) for point in box3D]

        if class_name not in self.object_settings["exported_object_classes"]:
            self.object_settings["exported_object_classes"].append(class_name)
            self.object_settings['exported_objects'].append({
                    "class": class_name,
                    "segmentation_class_id": class_id,
                    "cuboid_dimensions": self.objdim[class_name]
                    })
        self.data_dict = {
                "class": class_name,
                "class_id": class_id,
                "location":env_object.get_position(),
                "quaternion_xyzw": env_object.get_orientation(),
                "cuboid_centroid": cuboid_centroid,
                "projected_cuboid_centroid": projected_cuboid_centroid,
                "bounding_box": bounding_box,
                "cuboid": cuboid,
                "projected_cuboid": projected_cuboid,
                "box3D": box3D,
                "projected_3DBB": projected_3DBB,
                "projected_3DBB_centroid": projected_3DBB_centroid,
                }
        data["objects"].append(self.data_dict)

    def visualize(self): #DOPE
        image = im
        for projected_cuboid_point in data["objects"][-1]["projected_cuboid"]:
            image = cv2.circle(cv2.UMat(image), tuple(map(int, projected_cuboid_point)), 4, [0,255,0], -1)
        for projected_cuboid_point in data["objects"][-1]["projected_3DBB"]:
            image = cv2.circle(cv2.UMat(image), tuple(map(int, projected_cuboid_point)), 4, [255,0,0], -1)
        image = cv2.circle(cv2.UMat(image), tuple(map(int, [data["objects"][-1]["projected_cuboid_centroid"][0],data["objects"][-1]["projected_cuboid_centroid"][1]])), 4, [0,255,0], -1)
        image = cv2.circle(cv2.UMat(image), tuple(map(int, [data["objects"][-1]["projected_3DBB_centroid"][0],data["objects"][-1]["projected_3DBB_centroid"][1]])), 4, [255,0,0], -1)
        image = cv2.circle(cv2.UMat(image), tuple(map(int, [data["objects"][-1]["bounding_box"]["top_left"][0],data["objects"][-1]["bounding_box"]["top_left"][1]])), 4, [255,255,0], -1)
        image = cv2.circle(cv2.UMat(image), tuple(map(int, [data["objects"][-1]["bounding_box"]["bottom_right"][0],data["objects"][-1]["bounding_box"]["bottom_right"][1]])), 4, [255,255,0], -1)
        cv2.imshow('image',image)
        cv2.waitKey(1000)
        self.draw_bounding_box_3D()

    def draw_bounding_box_3D(self): #DOPE
        for points in range(7):
                #p.addUserDebugLine([0,0,0], [1,2,3], lineColorRGB=(0.31, 0.78, 0.47), lineWidth = 10)
                #points2=(points[0]+0.001,points[1]+0.001,points[2]+0.001)
                p.addUserDebugLine(data["objects"][-1]["box3D"][points],data["objects"][-1]["box3D"][points+1], lineColorRGB=(0.0, 0.0, 0.99), lineWidth = 4, lifeTime = 1)

    def write_json_end(self): #DOPE
        self.camera = {"camera_settings": []}
        for c in range(np.count_nonzero(config['active_cameras'])): #loop through active cameras
            camera_id=np.nonzero(config['active_cameras'])[0][c]
            intrinsic_settings = env.get_camera_opencv_matrix_values(camera_id)
            captured_image_size = {"width": config['camera_resolution'][0], "height": config['camera_resolution'][1]}
            self.camera["camera_settings"].append(dict(
                name="camera" + str(camera_id),
                intrinsic_settings=intrinsic_settings,
                captured_image_size=captured_image_size,
                ))
        if config['make_dataset'] in ["new", "resume"]:
            filename = "_camera_settings" + '.json'
            print("Storing {}.".format(filename))
            with open(os.path.join(config['output_test_folder'], filename), 'w') as f:
                json.dump(self.camera, f, indent=4)
            with open(os.path.join(config['output_train_folder'], filename), 'w') as f:
                json.dump(self.camera, f, indent=4)
            filename = "_object_settings" + '.json'
            with open(os.path.join(config['output_test_folder'], filename), 'w') as f:
                json.dump(self.object_settings, f, indent=4)
            with open(os.path.join(config['output_train_folder'], filename), 'w') as f:
                json.dump(self.object_settings, f, indent=4)


class GeneratorVae:
    """
    Generator class for image dataset for VAE vision model training
    """
    def __init__(self):
        self.object_settings = {"exported_object_classes": [], "exported_objects": []}
        self.env = None
        self.imsize = config["imsize"] # only supported format at the moment

    def get_env(self):
        """
        Create environment for VAE dataset generation according to dataset config file
        """
        self.env = RandomizedEnvWrapper(env=gym.make(config['env_name'],
            robot = config['robot'],
            render_on = True,
            gui_on = config['gui_on'],
            show_bounding_boxes_gui = config['show_bounding_boxes_gui'],
            changing_light_gui = config['changing_light_gui'],
            shadows_on = config['shadows_on'],
            color_dict = config['color_dict'],
            object_sampling_area = config['object_sampling_area'],
            observation = config["observation"],
            used_objects = used_objects,
            task_objects = config["task_objects"],
            active_cameras = config['active_cameras'],
            camera_resolution = config['camera_resolution'],
            dataset = True,
            ), config_path = config['output_folder']+'/config_dataset.json')
        p.setGravity(0, 0, -9.81)

    def collect_data(self, steps):
        """
        Collect data for VAE dataset

        Parameters:
            :param steps: (int) Number of episodes initiated during dataset generation
        """
        data = np.zeros((steps, self.imsize, self.imsize, 3), dtype='f')
        for t in range(steps):
            self.env.reset(random_pos=True)
            self.env.render()
            action = [random.uniform(1,2) for x in range(6)]
            self.env.robot.reset_random()
            # send the Kuka arms up
            observation, reward, done, info = self.env.step(action)
            img = observation['camera_data'][4]['image']
            imgs = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(imgs[0:450,100:500], (self.imsize, self.imsize))
            cv2.imshow("image", img)
            cv2.waitKey(1)
            padding = 6 - len(str(t+7999))
            name = padding * "0" + str(t+7999)
            cv2.imwrite(os.path.join(dataset_pth, "img_{}.png".format(name)), img)
            data[t] = img
            print("Image {}/{}".format(t, steps))
        self.env.close()

# main
if __name__ == "__main__":
    if len(sys.argv) <= 1:
        config_path = CONFIG_DEFAULT
        print('No config.json passed in argument. Loading default config: {}'.format(CONFIG_DEFAULT))
    else:
        config_path = pkg_resources.resource_filename("myGym", sys.argv[1])
    with open(config_path) as file:
        config = commentjson.load(file)

    # initialize dataset generator
    if config['dataset_type'] == 'coco':
        generator = GeneratorCoco()
    elif config['dataset_type'] == 'dope':
        generator = GeneratorDope()
    elif config['dataset_type'] == 'vae':
        generator = GeneratorVae()
    else:
        raise Exception("dataset_type in config: use one of 'coco', 'dope', 'vae'!")

    #prepare directories
    config['output_test_folder'] = config['output_folder'] + '/test'
    config['output_train_folder'] = config['output_folder'] + '/train'
    os.makedirs(config["output_test_folder"], exist_ok=True)
    os.makedirs(config["output_train_folder"], exist_ok=True)

    #define objects to appear in the env, add colors
    config['used_class_names'] = dict([x[1:3] for x in config['used_class_names_quantity']])
    used_objects = {"num_range":config["num_objects_range"], "obj_list":[]}
    for x in config['used_class_names_quantity']:
        for _ in range(x[0]):
            used_objects["obj_list"].append({"obj_name":x[1], "fixed":0,"rand_rot":1, "sampling_area":config["object_sampling_area"]})
    if config['color_dict'] == 'object_colors':
        color_names_to_rgb()
        config['texture_randomizer']['exclude'].append("objects")
        config['color_randomizer']['exclude'].append("objects")

    #write config.json to output_folder
    with open(config['output_folder']+'/config_dataset.json', 'w') as f:
          commentjson.dump(config, f)

    if config['dataset_type'] == 'vae':
            generator.get_env()
            dataset_pth = config['output_folder'] + '/train'
            generator.collect_data(config['num_episodes'])
            print("Dataset finished. Ready to train!")
            raise SystemExit(0)

    # initialize pybullet env
    env = generator.get_env()
    first_link_uid = env.robot.robot_uid
    robot_uids = np.array([((x + 1) << 24) + first_link_uid for x in range(-1, env.robot.gripper_index)],dtype=np.int32)
    gripper_uids = np.array([((x + 1) << 24) + first_link_uid for x in range(env.robot.gripper_index, env.robot.num_joints + 1)])

    # check mode of writing files
    image_id = 0 #for paralel dataset generation >0, otherwise 0
    if config['make_dataset'] == "new": #cleanup files
        files = glob.glob(os.path.join(config['output_test_folder'],'./*'))
        for f in files:
            os.remove(f)
        files = glob.glob(os.path.join(config['output_train_folder'],'./*'))
        for f in files:
            os.remove(f)
        data_train, data_test = generator.init_data()
        generator.episode_zero()
    elif config['make_dataset'] == 'resume': #resume
        print("Restoring dataset generation")
        data_test, data_train, image_id = generator.resume()
    elif (config['make_dataset'] == "display"):
        data_train, data_test = generator.init_data()
        generator.episode_zero()

    # the main loop
    for episode in range(int(image_id/(config['num_steps']*np.count_nonzero(config['active_cameras']))), config['num_episodes']): #loop through episodes
        print("episode: {}/{}".format(episode, config['num_episodes']))
        #env reset
        #random_robot randomizes the init position of robots
        #random_pos randomizes the init positions of objects
        # if episode == 0:
        #     generator.episode_zero()
        if episode % config['num_episodes_hard_reset'] == 0: #to prevent objects vanishing when GUI is on
            print("Hard reset!!!")
            env.reset(hard=True)
        env.reset(random_robot=config['random_arm_movement'], random_pos=True, hard=False)
        observation = env.get_observation()
        env_objects = observation["objects"]
        
        for t in range(config['num_steps']): #loop through steps
            # randomize the movements of robots (using joint control)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if t == config['num_steps']-1 or t%config['make_shot_every_frame'] == 0: # we only use frames from some steps
                env.render() #only render at the steps/frames we use for dataset
                for c in range(np.count_nonzero(config['active_cameras'])): #loop through active cameras
                    camera_id=np.nonzero(config['active_cameras'])[0][c]
                    image_id = image_id + 1 #unique
                    isTestSample = np.random.random_sample() < config['train_test_split_pct'] # bool, test/train data?
                    path = config['output_test_folder'] if isTestSample == True else config['output_train_folder']

                    #get dataset image and its mask
                    im = observation["camera_data"][camera_id]["image"]
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR) #fix colors
                    img_mask = observation["camera_data"][camera_id]["segmentation_mask"]
                    obj_ids = [x for x in np.unique(img_mask)] #identify objects(links) in the camera view (in the image)
                    img_mask = np.where(np.isin(img_mask,gripper_uids), gripper_uids[0], img_mask) #merge gripper links
                    img_mask = np.where(np.isin(img_mask,robot_uids), robot_uids[0], img_mask) #merge robot links
                    obj_ids = [x for x in np.unique(img_mask)] #identify merged objects in the camera view (in the image)
                    #prepare data strucuture
                    data, name = generator.data_struct_image()

                    for object_uid in obj_ids: #loop through kuka and used objects in the image (in the camera view)
                        if object_uid == robot_uids[0]:
                            class_name = config['robot']
                        elif object_uid == gripper_uids[0]:
                            class_name = env.robot.get_name()
                        else:
                            e = list(env_objects.values())
                            env_object_list = []
                            for sublist in e:
                                if isinstance(sublist, list):
                                    for item in sublist:
                                       env_object_list.append(item)
                                else:
                                    env_object_list.append(sublist)
                            env_objects_ids = [x.get_uid() for x in env_object_list]
                            if object_uid in env_objects_ids:
                                  env_object = env_object_list[env_objects_ids.index(object_uid)]
                                  class_name = env_object.get_name()
                            else:
                                continue
                        if class_name in config['used_class_names']:
                            class_id = config['used_class_names'][class_name]
                            generator.get_append_annotations() #annotate and append annotations
                            if config['visualize']: #visualize
                                generator.visualize()

                    #store dataset image and info
                    generator.store_image_info()
                    if config['make_dataset'] in ["new", "resume"]:
                        cv2.imwrite(os.path.join(path, name), im, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

        # write JSON annotations every n periods or at the end
        if episode % config['autosafe_episode'] == 0 or episode == config['num_episodes']-1:
            generator.write_json_end()
            data_train, data_test = generator.init_data()

    # end
    print('DATASET FINISHED')
