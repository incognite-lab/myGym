from myGym.envs import env_object, base_env
import numpy as np
import random

class DistractorModule():

    def __init__(self, distractor_moveable=0,
                 distractor_movement_endpoints=[-0.7+0.12, 0.7-0.15, 0, 0, 0, 0],
                 distractor_constant_speed=1, 
                 distractor_movement_dimensions=1, 
                 env=None,
                 ):

        self.distractor_moveable = distractor_moveable                       # the distcractor can move (on it's own)
        self.distractor_stopped  = False                                     # distractors movement was temporarily disabled
        self.distractor_movement_endpoints  = distractor_movement_endpoints  # coordinates of points between which is moving the distractor
        self.distractor_constant_speed      = distractor_constant_speed      # 1 => the speed is constant, 2 => the speed is not constant
        self.distractor_movement_dimensions = distractor_movement_dimensions # number of directions in which can distractor move
        self.goal_position = []                                              # position of object to reach (for meaningfull distractor placement)
        self.direction     = 1                                               # -1 => moves left, 1 => moves right (or the other way) 
        self.env           = env

        self.Xdirection     = 1                                              # -1 => moves left, 1 => moves right (or the other way) 
        self.Ydirection     = 1                                              # -1 => moves left, 1 => moves right (or the other way) 
        self.Zdirection     = 1                                              # -1 => moves left, 1 => moves right (or the other way) 


    def place_distractor(self, distractor, p):

        # position = "mezi rukou a cílem"
        # get position of gripper
        links = self.env.robot.get_links_observation(self.env.observed_links_num)
        # gripper_position = links[3*self.env.robot.gripper_index: (3+(3*self.env.robot.gripper_index))]
        # gripper_position = links[-4: -1] # last in observation
        gripper_position = [0.0, 0.4, 0.5] # last in observation
        # get position of goal
        # goal_position = self.env.goal_position
        goal_position = self.env.get_observation()[0:3]

        # get position in between
        a = 0.07 # constant needed, becouse goal spwans above table and then falls
        position = [((gripper_position[0]+goal_position[0])/2), ((gripper_position[1]+goal_position[1])/2), ((gripper_position[2]+goal_position[2])/2)-a]

        orientation = [0,0,50,50]
        object_filename = self.env._get_random_urdf_filenames(1, [distractor])[0]
        
        
        object = env_object.EnvObject(object_filename, position, orientation, pybullet_client=p)
        object.move([0,0,0]) # turn off gravity
        
        # if self.color_dict:
        #     object.set_color(self.env.color_of_object(object))

        return object

    def execute_distractor_step(self, name):
        for obj in self.env.env_objects:
            if obj.name == name: # if is object distractor
                if self.distractor_moveable and not self.distractor_stopped:
                    self.move_distractor(obj)
                else:
                    obj.move([0,0,0])

    def move_distractor(self, obj):
        # this is called every step, so the value actually doesnt represent movement, but speed

        if self.distractor_movement_dimensions == 1:
            self.move_1D(obj, not self.distractor_constant_speed)
        elif self.distractor_movement_dimensions == 2:
            self.move_2D(obj, not self.distractor_constant_speed)
        elif self.distractor_movement_dimensions == 3:
            self.move_3D(obj, not self.distractor_constant_speed)
        else:
            print("movement dimension of distractor makes no sense")
            exit()
  
    def move_1D(self, obj, chaotic):
        position = obj.get_position()[0]
        x = 0.003

        if chaotic:
            x = np.random.uniform(low=0.001, high=0.01)# * self.random_sign()

        if position < self.distractor_movement_endpoints[0] or position > self.distractor_movement_endpoints[1]:
            self.Xdirection = -self.Xdirection    

        if self.Xdirection > 0:
            obj.move([x,0,0])
        else:
            obj.move([-x,0,0])

    def move_2D(self, obj, chaotic):
        Xposition = obj.get_position()[0]
        Yposition = obj.get_position()[1]
        x = 0.003
        y = 0.003

        if chaotic:
            x = np.random.uniform(low=0.001, high=0.01)# * self.random_sign()
            y = np.random.uniform(low=0.001, high=0.01)# * self.random_sign()

        if Xposition < self.distractor_movement_endpoints[0] or Xposition > self.distractor_movement_endpoints[1]:
            self.Xdirection = -self.Xdirection

        if Yposition < self.distractor_movement_endpoints[2] or Yposition > self.distractor_movement_endpoints[3]:
            self.Ydirection = -self.Ydirection

        if self.Xdirection > 0:
            obj.move([x,0,0])
        else:
            obj.move([-x,0,0])
        if self.Ydirection > 0:
            obj.move([0,y,0])
        else:
            obj.move([0,-y,0])

    def move_3D(self, obj, chaotic):
        Xposition = obj.get_position()[0]
        Yposition = obj.get_position()[1]
        Zposition = obj.get_position()[2]
        x = 0.003
        y = 0.003
        z = 0.003

        if chaotic:
            x = np.random.uniform(low=0.001, high=0.01)# * self.random_sign()
            y = np.random.uniform(low=0.001, high=0.01)# * self.random_sign()
            z = np.random.uniform(low=0.001, high=0.01)# * self.random_sign()

        if Xposition < self.distractor_movement_endpoints[0] or Xposition > self.distractor_movement_endpoints[1]:
            self.Xdirection = -self.Xdirection

        if Yposition < self.distractor_movement_endpoints[2] or Yposition > self.distractor_movement_endpoints[3]:
            self.Ydirection = -self.Ydirection

        if Zposition < self.distractor_movement_endpoints[4] or Zposition > self.distractor_movement_endpoints[5]:
            self.Zdirection = -self.Zdirection

        if self.Xdirection > 0:
            obj.move([x,0,0])
        else:
            obj.move([-x,0,0])
        if self.Ydirection > 0:
            obj.move([0,y,0])
        else:
            obj.move([0,-y,0])
        if self.Zdirection > 0:
            obj.move([0,0,z])
        else:
            obj.move([0,0,-z])

    def random_sign(self):
        return 1 if random.random() < 0.5 else -1