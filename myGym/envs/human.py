import pkg_resources

from myGym.utils.helpers import get_robot_dict


class Human:
    """
    Human class for control of human environment interaction

    Parameters:
        :param model: (string) Model name in the get_robot_dict() dictionary
        :param pybullet_client: Which pybullet client the environment should refere to in case of parallel existence of multiple instances of this environment
    """
    def __init__(self,
                 model='human',
                 pybullet_client=None,
                 ):

        self.body_id = None
        self.n_joints = None
        self.motors_indices = []
        self.n_motors = None
        self.end_effector_idx = None
        self.p = pybullet_client

        self._load_robot(model)
        self._set_motors()

    def _load_robot(self, model):
        """
        Load SDF or URDF model of specified model and place it in the environment to specified position and orientation

        Parameters:
            :param model: (string) Model name in the get_robot_dict() dictionary
        """
        path, position, orientation = get_robot_dict()[model].values()
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
        Get overall description of the human. Mainly for debug.

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

    def point_finger_at(self, position):
        """
        Point human's finger towards the desired position.

        Parameters:
            :param position: (list) Cartesian coordinates [x,y,z]
        """
        self._run_motors(self._calculate_motor_poses(position))
