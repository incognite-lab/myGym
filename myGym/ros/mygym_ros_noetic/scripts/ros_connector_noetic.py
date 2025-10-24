#!/usr/bin/env python3
from zmq_comm.pub_sub import ParamSubscriber
import urwid
from mygym_ros_noetic.robot_proxy_noetic import robots, RobotProxy
from mygym_ros_noetic.action_stack import ActionStack
import argparse
import rospy
import traceback as tb
import PySimpleGUI as sg
from threading import Thread
from typing import Any


class RePublisher():
    PORT = 556677
    ADDRESS = "0.0.0.0"  # keep at 0.0.0.0 if local

    def __init__(self, robot_name: str, *robot_args, **robot_kwargs) -> None:
        rospy.init_node("mygym_robot_republisher")

        if robot_name in robots:
            self._robot_name = robot_name
            self._robot: RobotProxy = robots[self._robot_name](*robot_args, **robot_kwargs)
        else:
            raise ValueError(f"Unknown robot name {robot_name}!\nValid robots are {list(robots.keys())}.")

        self._subscriber = ParamSubscriber(start_port=self.PORT, addr=self.ADDRESS)
        self._subscriber.set_callback(self._cb)
        self._subscriber.subscribe("robot_action")
        self._subscriber.subscribe("robot_grip")

        self._stack = ActionStack(self._robot.joint_space_dim)

        self._update_msg_count = False

    def __get_ui_elem(self, elem_name) -> Any:
        res = self._ui_window[elem_name]
        if res is sg.ErrorElement:
            raise ValueError(f"Element {elem_name} does not exist!")
        else:
            return res

    def _construct_ui(self):
        layout = [
            [sg.Text(f"Robot name: {self._robot_name}")],
            [
                sg.ProgressBar(0, orientation='h', size=(100, 20), key='msg_progress'),
                sg.Text("what the fridge?", key="msg_count")
            ],
            # [sg.Text("Not connected to myGym transmitter.", key="label_transmitter_conn")],
        ]

        self._ui_window = sg.Window(
            layout=layout,
            title="MyGym sim to ROS Noetic robot trajectory republisher",
            finalize=True
        )

    def startup(self):
        self._construct_ui()

        th = Thread(target=self._run_thread, daemon=True)
        th.start()
        self._run_ui()

    def _run_ui(self):
        msg_progress: sg.ProgressBar = self.__get_ui_elem("msg_progress")
        msg_count: sg.Text = self.__get_ui_elem("msg_count")
        # label_transmitter_conn = self.__get_ui_elem"label_transmitter_conn")

        def update_msg_count():
            msg_count.update(f"{len(self._stack)} msgs received")
            msg_progress.update_bar(0, max=len(self._stack))

        # update_msg_count()
        while not rospy.is_shutdown():
            try:
                res = self._ui_window.read(timeout=10)
                if res is not None:
                    event, values = res
                    if event == sg.WIN_CLOSED:
                        break
                    # else:
                    #     pass
                if self._update_msg_count:
                    print("Updating msg count.")
                    update_msg_count()
                    self._update_msg_count = False
            except BaseException as e:
                rospy.logerr(f"Failed to update UI: {e}")
                raise e

    def _run_thread(self):
        rospy.loginfo("Started spinning thread.")
        try:
            rospy.spin()
        except BaseException as e:
            rospy.signal_shutdown("Error!")
            raise e

    def _cb(self, param, msg):
        self._update_msg_count = True
        if param == "robot_action":
            self._append_action(msg)
        elif param == "robot_grip":
            self._append_grip(msg)
        else:
            raise ValueError(f"Somehow got an unknown type of robot operation: {param} with parameters: {msg}!")

    def _append_action(self, action_msg):
        rospy.loginfo("* trajectory point received")
        self._stack.append_trajectory(action_msg)

    def _append_grip(self, grip_msg):
        rospy.loginfo("* gripper operation received")
        self._stack.grip(grip_msg)

    def destroy(self):
        self._subscriber.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-name", "--robot", "-r", default="panda", type=str)
    parser.add_argument("--robot-args", "--args", "-a", action="append")

    args = parser.parse_args()

    repub = RePublisher(args.robot_name, args.robot_args)

    try:
        repub.startup()
    except KeyboardInterrupt:
        rospy.signal_shutdown("User interrupt.")
        print("Interrupted by user.")
    finally:
        print("Shutting down.")
        repub.destroy()
