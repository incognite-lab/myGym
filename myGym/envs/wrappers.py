import gym
import commentjson


from myGym.envs import randomizers


class RandomizedEnvWrapper(gym.Wrapper):
    """
    Wrapper for the environment which need to be randomized

    Parameters:
        :param env: (BaseEnv) Environment to be randomized
        :param config_path: (str) Path to the config file
    """

    def __init__(self, env, config_path):
        super(RandomizedEnvWrapper, self).__init__(env)
        self.config_path = config_path
        with open(config_path) as file:
            self.config = commentjson.load(file)
        self.active_randomizers = []
        if "texture_randomizer" in self.config:
            self.active_randomizers.append(randomizers.TextureRandomizer(env, self.config["seed"], **self.config["texture_randomizer"]))
        if "light_randomizer" in self.config:
            self.active_randomizers.append(randomizers.LightRandomizer(env, self.config["seed"], **self.config["light_randomizer"]))
        if "camera_randomizer" in self.config:
            self.active_randomizers.append(randomizers.CameraRandomizer(env, self.config["seed"], **self.config["camera_randomizer"]))
        if "color_randomizer" in self.config:
            self.active_randomizers.append(randomizers.ColorRandomizer(env, self.config["seed"], **self.config["color_randomizer"]))
        if "joint_randomizer" in self.config:
            self.active_randomizers.append(randomizers.JointRandomizer(env, self.config["seed"], **self.config["joint_randomizer"]))

    def randomize(self):
        """
        Randomize env using active randomizers
        """
        for active_randomizer in self.active_randomizers:
            if active_randomizer.is_enabled():
                active_randomizer.randomize()

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        """
        Randomize with each reset
        """
        observation = self.env.reset(**kwargs)
        self.randomize()
        return observation

    def hard_reset(self, **kwargs):
        observation = self.env.hard_reset(**kwargs)
        return observation

    def render(self, **kwargs):
        return self.env.render(**kwargs)
