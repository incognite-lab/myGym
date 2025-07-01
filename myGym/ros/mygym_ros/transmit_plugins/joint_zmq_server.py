#!/usr/bin/env python2
"""
ROS node to send joint trajectories to TIAGo's arm via action client.
Supports left or right arm, defaulting to left. Safe shutdown cancels goals.
Includes test_joint_action to send a random perturbation trajectory.
"""
from __future__ import print_function
import rospy
import actionlib
import argparse
import math
import random
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, FollowJointTrajectoryResult, FollowJointTrajectoryFeedback
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from actionlib_msgs.msg import GoalStatus
import zmq
from threading import Thread
from warnings import warn
import cloudpickle as cpl
from zmq.utils.strtypes import asbytes


class ServiceServer():
    SEND_TIMEOUT = 3000  # in ms (how long to wait to send the request - doesn't do much...)
    RECEIVE_TIMEOUT = 3000  # in ms (how long to wait to receive a response)

    def __init__(self, callback, port=242424, addr="0.0.0.0", protocol="tcp"):
        """Create a service server. Callback should handle the incoming requests.
        A request will be a dictionary with some data (has to be agreed upon externally).
        Port should serve as a service identifier (if more services are used).

        Args:
            callback (function): Request handler.
            port (int, optional): This is the main service identifier. Defaults to 242424.
            addr (str, optional): Address of the service server. Defaults to "127.0.0.1".
            protocol (str, optional): Protocol to use, keep on default. Defaults to "tcp".
        """
        self.__context = zmq.Context()
        self.__addr = "{protocol}://{addr}:{port}".format(protocol=protocol, addr=addr, port=str(port))   # full address ~ sort of like a service name/identifier
        print("Creating service on {} (if localhost/loopback address is used, service will be visible to localhost only)".format(self.__addr))  # use actual ip addr for network visibility
        # bind the ZMQ socket
        self._connect()

        self._callback = callback
        self.__active = True
        # thread to wait for requests
        self.poller_thread = Thread(target=self.__poll)
        self.poller_thread.daemon = True
        self.poller_thread.start()

    def _reconnect(self):
        """Reconnect after error (e.g., service timeout) otherwise socket in weird state = will not work
        """
        print("Someone messed up and I had to reconnect the ZMQ socket!")
        self._zmq_socket.close(self.__addr)
        self._connect()

    def _connect(self):
        self._zmq_socket = self.__context.socket(zmq.REP)
        self._zmq_socket.setsockopt(zmq.SNDTIMEO, self.SEND_TIMEOUT)
        self._zmq_socket.setsockopt(zmq.RCVTIMEO, self.RECEIVE_TIMEOUT)
        self._zmq_socket.bind(self.__addr)

    def destroy(self):
        self.__active = False
        self.poller_thread.join()
        self._zmq_socket.close()

    @staticmethod
    def _convert_dict_to_unicode(indict):
        return {asbytes(k): v for k, v in indict.items()}

    def __poll(self):
        while self.__active:
            try:
                request = self._zmq_socket.recv()  # wait for a request
            except zmq.Again:
                continue
            # unpickle and send to callback
            # request_dict = cpl.loads(request, encoding="latin1")
            request_dict = cpl.loads(request)
            try:
                response_dict = self._callback(request_dict)
            except Exception as e:
                # if the callback rises unhandled error, send empty dict
                print("Error in the service callback:\n{}".format(e))
                response_dict = {}
            # pickle the response and send back to the caller
            # response_dict = self._convert_dict_to_unicode(response_dict)
            response = cpl.dumps(response_dict, protocol=2)
            try:
                self._zmq_socket.send(response)
            except zmq.Again:
                self._reconnect()


class ArmTrajectoryController(object):
    def __init__(self, side, unsafe=False):
        # Initialize ROS node
        rospy.init_node('arm_trajectory_controller', anonymous=True)
        self.side = side
        # Define joint names for TIAGo arm
        # TIAGo arms have 7 joints: arm_left_1_joint ... arm_left_7_joint
        self.joint_names = ['arm_{}_{}_joint'.format(side, i) for i in range(1, 8)]

        # Storage for current joint states
        self.joint_state = None
        rospy.Subscriber('/joint_states', JointState, self.joint_state_cb)

        # Action client for follow_joint_trajectory
        action_server = '/{}arm_{}_controller/follow_joint_trajectory'.format('safe_' if not unsafe else '', side)
        self.client = actionlib.SimpleActionClient(action_server, FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for action server: {}".format(action_server))
        # self.client.wait_for_server()
        rospy.loginfo("Connected to action server")

        # Ensure safe shutdown: cancel goal on exit
        rospy.on_shutdown(self.shutdown_hook)

        self.zmq_service_server = ServiceServer(self._data_callback)

        self.goal = None

    def _data_callback(self, data):
        joint_names = data['joint_names']
        joint_positions = data['data']

        if 'left' in joint_names[0]:
            arm_side = 'left'
        elif 'right' in joint_names[0]:
            arm_side = 'right'
        else:
            rospy.logerr("Invalid joint names: {}".format(joint_names))
            return

        for i, joint_name in enumerate(joint_names):
            if arm_side not in joint_name:
                rospy.logerr("Invalid joint name: {}".format(joint_name))
                return
            joint_name = joint_name.encode('ascii')
            joint_name = joint_name.replace('_rjoint', '_joint')
            # remove starting "b'"
            joint_name = joint_name.replace("b'", '')
            joint_name = joint_name.replace("'", '')
            joint_names[i] = joint_name

        print(joint_names)

        self.send_joint_action(joint_names, joint_positions, 5.0)

    def joint_state_cb(self, msg):
        # Store the latest joint state
        self.joint_state = msg

    def shutdown_hook(self):
        rospy.loginfo("Shutting down: cancelling any active goal...")
        if self.goal is not None:
            self.client.cancel_goal(self.goal)
        self.client.cancel_all_goals()
        # Wait briefly for cancellation confirmation
        finished_before_timeout = self.client.wait_for_result(rospy.Duration(5.0))
        if finished_before_timeout:
            state = self.client.get_state()
            rospy.loginfo("Goal cancelled, final state: %d" % state)
        else:
            rospy.logwarn("Cancellation not confirmed within timeout")

    def _wait_for_joint_states(self):
        # Wait until we have joint states
        timeout = rospy.Time.now() + rospy.Duration(5.0)
        while self.joint_state is None and not rospy.is_shutdown():
            if rospy.Time.now() > timeout:
                rospy.logerr("Timed out waiting for joint_states")
                return False
            rospy.sleep(0.1)
        return True

    def get_current_positions(self, joint_names=None):
        if joint_names is None:
            joint_names = self.joint_names

        if not self._wait_for_joint_states():
            return None

        # Map joint names to positions
        name_to_pos = dict(zip(self.joint_state.name, self.joint_state.position))
        rospy.loginfo("Current positions: {}".format(name_to_pos))
        # Extract positions in order
        try:
            return [name_to_pos[name] for name in joint_names]
        except KeyError as e:
            rospy.logerr("Joint name missing in joint_states: %s" % e)
            return None

    def feedback_cb(self, feedback):
        # rospy.loginfo_throttle(1.0, "Feedback received: time_from_start=%.2f" % (
        #     feedback.actual.time_from_start.to_sec()))
        rospy.loginfo_throttle(1.0, "Feedback received:\n{}\n{}".format(feedback.actual.positions, feedback.actual.velocities))


    def send_joint_action(self, joint_names, positions, durations=None):
        """
        Send a trajectory to TIAGo's arm via action client.
        """

        init_positions = self.get_current_positions(joint_names)
        if init_positions is None:
            rospy.logerr("The input joint names {} do not match the current joint states".format(joint_names))
            return
        rospy.loginfo("Current positions: {}".format(init_positions))

        if durations is None:
            durations = [(i + 1) * 1.0 for i in range(len(positions))]
        elif isinstance(durations, float):
            durations = [(i + 1) * durations for i in range(len(positions))]
        else:
            assert len(durations) == len(positions)
            durations = list(durations)

        # Build trajectory
        traj = JointTrajectory()
        traj.joint_names = joint_names
        points = []

        for pos, dur in zip(positions, durations):
            pt = JointTrajectoryPoint()
            pt.positions = pos
            # simple timing: 1 second apart
            pt.time_from_start = rospy.Duration(dur)
            points.append(pt)

        traj.points = points
        self.goal = FollowJointTrajectoryGoal()
        self.goal.trajectory = traj
        self.goal.trajectory.header.stamp = rospy.Time(0)

        rospy.loginfo("Sending test goal:")
        rospy.loginfo(str(self.goal))

        # Send goal
        self.client.send_goal(self.goal, feedback_cb=self.feedback_cb)
        rospy.loginfo("Goal sent, waiting for result...")
        self.client.wait_for_result(rospy.Duration(100.0))
        state = self.client.get_state()
        rospy.loginfo("Action completed with state: %d" % state)
        self.goal = None


    def test_joint_action(self, num_points, max_deg):
        """
        Send a test trajectory of num_points, where each joint is perturbed
        by at most max_deg degrees from previous point.
        """
        rospy.loginfo("Generating test trajectory: %d points, max %.2f deg" % (num_points, max_deg))
        # Get current joint positions
        init_positions = self.get_current_positions()
        if init_positions is None:
            return
        max_rad = math.radians(max_deg)

        # Build trajectory
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        points = []
        prev_positions = init_positions

        for i in range(num_points):
            new_positions = []
            for p in prev_positions:
                new_positions.append(p + random.uniform(-max_rad, max_rad))
            pt = JointTrajectoryPoint()
            pt.positions = new_positions
            # simple timing: 1 second apart
            pt.time_from_start = rospy.Duration((i + 1) * 1.0)
            points.append(pt)
            prev_positions = new_positions

        traj.points = points
        self.goal = FollowJointTrajectoryGoal()
        self.goal.trajectory = traj
        # Stamp a little in the future
        # goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(0.1)
        # goal.trajectory.header.stamp = rospy.Time.now()
        self.goal.trajectory.header.stamp = rospy.Time(0)

        rospy.loginfo("Sending test goal:")
        rospy.loginfo(str(self.goal))

        # Send goal
        self.client.send_goal(self.goal, feedback_cb=self.feedback_cb)
        rospy.loginfo("Test goal sent, waiting for result...")
        self.client.wait_for_result(rospy.Duration(30.0))
        state = self.client.get_state()
        rospy.loginfo("Test action completed with state: %d" % state)
        self.goal = None


def main():
    parser = argparse.ArgumentParser(description='TIAGo arm trajectory controller')
    parser.add_argument('--side', choices=['left', 'right'], default='left',
                        help='Arm side to control (left or right)')
    parser.add_argument('--test', nargs=2, type=float, metavar=('NUM_POINTS', 'MAX_DEG'),
                        help='Run test_joint_action with NUM_POINTS and MAX_DEG degrees')
    parser.add_argument('--unsafe', action='store_true', help='Using the "unsafe" action client')
    args = parser.parse_args()

    controller = ArmTrajectoryController(args.side, args.unsafe)

    if args.test:
        num_points, max_deg = int(args.test[0]), args.test[1]
        rospy.sleep(1.0)  # wait for subscribers
        controller.test_joint_action(num_points, max_deg)
    else:
        rospy.loginfo("Ready to send trajectories. Use controller.test_joint_action() in code or extend this script.")
        rospy.spin()

if __name__ == '__main__':
    main()
