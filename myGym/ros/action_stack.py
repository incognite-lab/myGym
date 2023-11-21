from collections import deque
from typing import Literal, Any
import numpy.typing as npt
import numpy as np


OPERATION_TYPES = Literal['trajectory', 'close_gripper', 'open_gripper']


class Operation():
    TYPE_TRAJECTORY: OPERATION_TYPES = 'trajectory'
    TYPE_CLOSE_GRIPPER: OPERATION_TYPES = 'close_gripper'
    TYPE_OPEN_GRIPPER: OPERATION_TYPES = 'open_gripper'

    def __init__(self, typ: OPERATION_TYPES) -> None:
        self._type = typ

    @property
    def type(self):
        return self._type


class Trajectory(Operation):

    def __init__(self, dimensionality: int = 7) -> None:
        super().__init__(Operation.TYPE_TRAJECTORY)

        self._queue: deque[npt.ArrayLike] = deque()
        self._expected_size = dimensionality

    def append_point(self, point_pose: npt.ArrayLike):
        point_pose = np.asanyarray(point_pose)
        if point_pose.size != self._expected_size:
            raise ValueError(f"Received trajectory pose of incorrect size! Expected dimensionality is {self._expected_size} but it was {point_pose.size}!\nThe pose was {point_pose}")
        self._queue.append(point_pose)


class CloseGripper(Operation):

    def __init__(self) -> None:
        super().__init__(Operation.TYPE_CLOSE_GRIPPER)


class OpenGripper(Operation):

    def __init__(self) -> None:
        super().__init__(Operation.TYPE_OPEN_GRIPPER)


class ActionStack():

    def __init__(self, trajectory_dimensionality: int = 7) -> None:
        self._queue: deque[Operation] = deque()
        self._expected_trajectory_size = trajectory_dimensionality

    def append_trajectory(self, point_pose: npt.ArrayLike):
        if len(self._queue) == 0 or self._queue[-1].type != Operation.TYPE_TRAJECTORY:
            self._queue.append(Trajectory(self._expected_trajectory_size))

        self._queue[-1].append_point(point_pose)

    def grip(self, open: bool):
        if open:
            self._queue.append(OpenGripper())
        else:
            self._queue.append(CloseGripper())

    def __iter__(self):
        return iter(self._queue)

    def __len__(self):
        return len(self._queue)
