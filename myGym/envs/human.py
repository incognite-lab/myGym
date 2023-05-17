from typing import List, Tuple

import numpy as np
import pkg_resources

from myGym.envs.env_object import EnvObject
from myGym.utils.helpers import get_robot_dict


def _link_name_to_idx(p, body_id: int, link_name: str) -> int:
    for i in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, i)
        if info[-5].decode('utf-8') == link_name:
            return i
    exc = f"Cannot find a link index for the link with name {link_name}"
    raise Exception(exc)


class Human:
    """
    Class for control of human-environment interaction.

    Parameters:
        :param model_name: (string) Model name from the get_robot_dict() dictionary
        :param pybullet_client: Which pybullet client the environment should refer to in case of parallel existence
        of multiple instances of this environment
    """
    def __init__(self,
                 model_name: str = "human",
                 pybullet_client=None,
                 direction_point: np.array = np.array([0, 0.8, 0]),
                 links_for_direction_vector: Tuple[str, str] = ("r_index3_endeffector", "r_index2")
                 ):

        self.p = pybullet_client
        self.body_id: int or None = None
        self.n_joints: int or None = None
        self.motors_indices = []
        self.n_motors: int or None = None
        self.end_effector_idx: int or None = None
        self.direction_point = direction_point  # for pointing during a testing phase

        self._load_model(model_name)
        self._set_motors()

        self.links_indices_for_direction_vector = (
            _link_name_to_idx(self.p, self.body_id, links_for_direction_vector[0]),
            _link_name_to_idx(self.p, self.body_id, links_for_direction_vector[1])
        )

    def _load_model(self, model_name):
        """
        Load SDF or URDF model of specified model and place it in the environment to specified position and orientation.

        Parameters:
            :param model_name: (string) Model name in the get_robot_dict() dictionary
        """
        path, position, orientation = get_robot_dict()[model_name].values()
        path = pkg_resources.resource_filename("myGym", path)
        orientation = self.p.getQuaternionFromEuler(orientation)

        if path[-3:] == 'sdf':
            self.body_id = self.p.loadSDF(path)[0]
            self.p.resetBasePositionAndOrientation(self.body_id, position, orientation)
        else:
            self.body_id = self.p.loadURDF(path, position, orientation, useFixedBase=True, flags=(self.p.URDF_USE_SELF_COLLISION))

        self.n_joints = self.p.getNumJoints(self.body_id)
        for jid in range(self.n_joints):
            self.p.changeDynamics(self.body_id, jid, collisionMargin=0., contactProcessingThreshold=0.0, ccdSweptSphereRadius=0)

    def _set_motors(self):
        """
        Identify motors among all joints (fixed joints aren't motors).
        Identify index of end-effector link among all links. Uses data from human model.
        """
        for i in range(self.n_joints):
            info = self.p.getJointInfo(self.body_id, i)
            q_index = info[3]
            link_name = info[12]

            if q_index > -1:
                self.motors_indices.append(i)

            if 'endeffector' in link_name.decode('utf-8'):
                self.end_effector_idx = i

        self.n_motors = len(self.motors_indices)

        if self.end_effector_idx is None:
            print("No end effector detected. "
                  "Please define which link is an end effector by adding 'endeffector' to the name of the link")
            exit()

    def __repr__(self):
        """
        Get overall description of the human. Used mainly for debug.

        Returns:
            :return description: (string) Overall description
        """
        params = {'Id': self.body_id,
                  'Number of joints': self.n_joints,
                  'Number of motors': self.n_motors}
        description = 'Human parameters\n' + '\n'.join([k + ': ' + str(v) for k, v in params.items()])
        return description

    def _run_motors(self, motor_poses):
        """
        Move joint motors towards desired joint poses respecting model's dynamics

        Parameters:
            :param motor_poses: (list) Desired poses of individual joints
        """
        self.p.setJointMotorControlArray(self.body_id,
                                         self.motors_indices,
                                         self.p.POSITION_CONTROL,
                                         motor_poses,
                                         )

    def _calculate_motor_poses(self, end_effector_pos):
        """
        Calculate motor poses corresponding to desired position of end-effector. Uses inverse kinematics.

        Parameters:
            :param end_effector_pos: (list) Desired position of end-effector in the environment [x,y,z]
        Returns:
            :return motor_poses: (list) Calculated motor poses corresponding to the desired end-effector position
        """
        return self.p.calculateInverseKinematics(self.body_id,
                                                 self.end_effector_idx,
                                                 end_effector_pos,
                                                 )

    def point_finger_at(self, position=None, relative=False):
        """
        Point human's finger towards the desired position.

        Parameters:
            :param position: (list) Cartesian coordinates [x,y,z]
        """
        if relative:
            if position is not None:
                self.direction_point += position
                position = self.direction_point
            else:
                exc = "You must pass relative coordinates if a relative option has been chosen"
                raise Exception(exc)
        else:
            if position is None:
                position = self.direction_point
        self._run_motors(self._calculate_motor_poses(position))

    def find_object_human_is_pointing_at(self, objects: List[EnvObject]) -> EnvObject:
        if not objects:
            raise Exception("There are no objects!")

        i1, i2 = self.links_indices_for_direction_vector
        p1, p2 = self.p.getLinkState(self.body_id, i1)[0], self.p.getLinkState(self.body_id, i2)[0]
        p1, p2 = np.array(p1), np.array(p2)
        vec = (p1 - p2) / np.linalg.norm(p1 - p2)
        points = np.array([o.get_position() for o in objects])

        points -= p2.reshape(1, -1)  # move points to be able to compute projections (make the vector relatively centered)
        scalars = np.dot(points, vec)  # scalar product (as a part of computing a projection)
        points_proj = scalars.reshape(-1, 1) * vec.reshape(1, -1)  # projections on the vector
        points_rej = points - points_proj  # rejections
        distances = np.linalg.norm(points_rej, axis=1)
        return objects[np.argmin(distances)]
