
import numpy as np
import os, glob, random
import pybullet as p
import pkg_resources
repodir = pkg_resources.resource_filename("myGym", "")



class Randomizer:
    """
    Abstract class for randomizers

    Parameters:
        :param env: (BaseEnv) Environment to randomize
        :param seed: (int)
        :param enabled: (bool) Whether randomizer turned on or not
    """

    def __init__(self, env, seed, enabled):
        self.env = env
        self.seed = seed
        self.enabled = enabled
        np.random.seed(seed)

    def randomize(self):
        raise NotImplementedError

    def is_enabled(self):
        return self.enabled


class LightRandomizer(Randomizer):
    """
    Class for the light randomization

    Parameters:
        :param env: (BaseEnv) Environment to randomize
        :param seed: (int)
        :param enabled: (bool) Whether randomizer turned on or not
        :param randomized_dimensions: (list) Parameters of light to randomize
    """

    def __init__(self, env, seed, enabled, randomized_dimensions):
        super(LightRandomizer, self).__init__(env, seed, enabled)
        self.dimensions = {
            "light_direction": Dimension(default_value=[1, 1, 1], name="light_direction"),
            "light_color": Dimension(default_value=[1, 0, 0], name="light_color"),
            "light_distance": Dimension(default_value=1, name="light_distance"),
            "light_ambient": Dimension(default_value=1, name="light_ambient"),
            "light_diffuse": Dimension(default_value=1, name="light_diffuse"),
            "light_specular": Dimension(default_value=1, name="light_specular")
        }
        self.randomized_dimensions = randomized_dimensions

    def randomize(self):
        new_values = {}
        for dimension, is_enabled in self.randomized_dimensions.items():
            if is_enabled:
                new_values[dimension] = self.dimensions[dimension].randomize()
        self.env.set_light(**new_values)


class CameraRandomizer(Randomizer):
    """
    Class for the camera randomization

    Parameters:
        :param env: (BaseEnv) Environment to randomize
        :param seed: (int)
        :param enabled: (bool) Whether randomizer turned on or not
        :param randomized_dimensions: (list) Parameters of camera to randomize
        :param shift: (list) Maximum shift of camera position
    """

    def __init__(self, env, seed, enabled, randomized_dimensions, shift):
        super(CameraRandomizer, self).__init__(env, seed, enabled)
        self.cameras_dimensions = {}
        for camera in env.get_cameras():
            self.cameras_dimensions[camera] = {
                "target_position": Dimension(default_value=camera.target_position, name="target_position", shift=shift)
            }
        self.randomized_dimensions = randomized_dimensions
        self.shift = shift

    def randomize(self):
        for camera, dimensions in self.cameras_dimensions.items():
            new_values = {}
            for dimension, is_enabled in self.randomized_dimensions.items():
                if is_enabled:
                    new_values[dimension] = dimensions[dimension].randomize()
            camera.set_parameters(**new_values)


class TextureRandomizer(Randomizer):
    """
    Class for the texture randomization

    Parameters:
        :param env: (BaseEnv) Environment to randomize
        :param seed: (int)
        :param enabled: (bool) Whether randomizer turned on or not
        :param seamless: (bool) Use seamless textures
        :param textures_path: (str) Path to the directory with textures
        :param exclude: (str) Do not apply textures to excluded objects
    """

    def __init__(self, env, seed, enabled, seamless, textures_path, seamless_textures_path, exclude):
        super(TextureRandomizer, self).__init__(env, seed, enabled)
        self.textures_path = textures_path
        #cache
        self.all_textures_ = None
        self.exclude = exclude
        if seamless == True:
            textures_path = seamless_textures_path
        if textures_path is not None:
          pp = os.path.abspath(os.path.join(repodir, str(textures_path)))
          self.all_textures_ = glob.glob(os.path.join(pp, '**', '*.jpg'), recursive=True)

    def randomize(self):
        if 'objects' in self.exclude:
            textures_obj_ids = self.env.scene_objects_uids
        else:
            textures_obj_ids = self.env.get_texturizable_objects_uids()
        for object_id in textures_obj_ids:
            if object_id in self.env.scene_objects_uids.keys() and self.env.scene_objects_uids[object_id] in self.exclude:
                pass
            else:
                self.apply_texture(object_id, self.textures_path)

    def apply_texture(self, obj_id, patternPath="envs/dtd/images"):
        """
        Apply texture to pybullet object

        Parameters:
            :param obj_id: (int) ID obtained from `p.loadURDF/SDF/..()`
            :param path: (str) Relative path to *.jpg (recursive) with textures
        """
        if patternPath is None:
            return

        if patternPath == self.textures_path: # no change, can use cached value
          texture_paths = self.all_textures_
        else:
          pp = os.path.abspath(os.path.join(repodir, str(patternPath)))
          texture_paths = glob.glob(os.path.join(pp, '**', '*.jpg'), recursive=True)
        random_texture_path = texture_paths[random.randint(0, len(texture_paths) - 1)]
        textureId = p.loadTexture(random_texture_path)
        try:
            p.changeVisualShape(obj_id, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=textureId)
        except:
            print("Failed to apply texture to obj ID:"+str(obj_id)+" from path="+str(pp))


class ColorRandomizer(Randomizer):
    """
    Class for the color randomization

    Parameters:
        :param env: (BaseEnv) Environment to randomize
        :param seed: (int)
        :param enabled: (bool) Whether randomizer turned on or not
        :param randomized_dimensions: (list) Parameters of color to randomize
        :param exclude: (str) Do not apply color randomization to excluded objects
    """

    def __init__(self, env, seed, enabled, randomized_dimensions, exclude):
        super(ColorRandomizer, self).__init__(env, seed, enabled)
        self.dimensions = {
            "rgb_color": Dimension(default_value=[0, 0, 0], name="rgb_color"),
            "specular_color": Dimension(default_value=[1, 1, 1], multiplier_max=1, name="specular_color")
        }
        self.randomized_dimensions = randomized_dimensions
        self.exclude = exclude

    def randomize(self):
        if 'objects' in self.exclude:
            textures_obj_ids = self.env.scene_objects_uids
        else:
            textures_obj_ids = self.env.get_texturizable_objects_uids()
        for object_id in textures_obj_ids:
            if object_id in self.env.scene_objects_uids.keys() and self.env.scene_objects_uids[object_id] in self.exclude:
                pass
            else:
                if "rgb_color" in self.randomized_dimensions and self.randomized_dimensions["rgb_color"]:
                    new_value = self.dimensions["rgb_color"].randomize()
                    p.changeVisualShape(object_id, -1, rgbaColor=np.append(new_value, 1))
                if "specular_color" in self.randomized_dimensions and self.randomized_dimensions["specular_color"]:
                    new_value = self.dimensions["specular_color"].randomize()
                    p.changeVisualShape(object_id, -1, specularColor=new_value)


class JointRandomizer(Randomizer):

    def __init__(self, env, seed, enabled):
        super(JointRandomizer, self).__init__(env, seed, enabled)

    def randomize(self):
        textures_obj_ids = self.env.get_texturizable_objects_uids()
        for object_id in textures_obj_ids:
            self.random_reset_joints(object_id)

    def random_reset_joints(self, object_id):
        num_joints = p.getNumJoints(object_id)
        #joint_poses = list(np.random.uniform(0, 2*3.14, num_joints))
        for jid in range(num_joints):
            joint_limits = p.getJointInfo(object_id, jid)[8:10]
            joint_pose = np.random.uniform(joint_limits[0], joint_limits[1])
            # To apply ignore dynamics:
            p.resetJointState(object_id, jid, joint_pose)
            # p.setJointMotorControl2(object_id,
            #                         jid,
            #                         p.POSITION_CONTROL,
            #                         targetPosition=joint_pose,
            #                         force=200)





class PostprocessingRandomizer(Randomizer):

    def __init__(self, default_value, multiplier_min=0.0, multiplier_max=1.0, name=None):
        super(PostprocessingRandomizer, self).__init__(default_value, multiplier_min, multiplier_max, name)


class ObjectPositionsRandomizer(Randomizer):

    def __init__(self, default_value, multiplier_min=0.0, multiplier_max=1.0, name=None):
        super(ObjectPositionsRandomizer, self).__init__(default_value, multiplier_min, multiplier_max, name)


class Dimension:
    """
    Class representing 1 dimension to randomize

    Parameters:
        :param defaulet_value: (int or list) Initial value of dimension
        :param multiplier_min: (float) Min multiplier for randomization
        :param multiplier_max: (float) Max multiplier for randomization
        :param name: (str) Name of dimension
        :param shift: (float) Max possible shift from the default value
        :param seed: (int)
    """

    def __init__(self, default_value=None, multiplier_min=0.0, multiplier_max=1.0, name=None, shift=None, seed=None):
        self.default_value = default_value if type(default_value) is list else [default_value]
        self.subdimensions = len(self.default_value)
        self.current_value = self.default_value
        self.multiplier_min = multiplier_min
        self.multiplier_max = multiplier_max
        if shift:
            self.range_min = np.subtract(default_value, shift)
            self.range_max = np.add(default_value, shift)
        else:
            self.range_min = [0 * multiplier_min] * self.subdimensions
            self.range_max = [1 * multiplier_max] * self.subdimensions
        self.name = name
        self.seed = seed

    def randomize(self):
        """
        Set random value
        """
        self.current_value = np.random.uniform(low=self.range_min, high=self.range_max)
        #print("New value: {} {}".format(self.name, self.current_value))
        return self.current_value[0] if self.subdimensions == 1 else self.current_value

    def rescale(self, value):
        raise NotImplementedError

    def reset(self):
        self.current_value = self.default_value
        return self.current_value

    def set(self, value):
        self.current_value = value
