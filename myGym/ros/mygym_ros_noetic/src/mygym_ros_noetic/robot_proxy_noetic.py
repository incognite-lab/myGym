from typing import Dict


class RobotProxy():

    def __init__(self, joint_space_dim) -> None:
        self._joint_space_dim = joint_space_dim

    def perform_trajectory(self, trajectory):
        raise NotImplementedError()

    def close_gripper(self):
        raise NotImplementedError()

    def open_gripper(self):
        raise NotImplementedError()

    @property
    def joint_space_dim(self) -> int:
        return self._joint_space_dim


class PandaProxy(RobotProxy):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(7)


robots: Dict[str, type] = {
    "panda": PandaProxy
}
