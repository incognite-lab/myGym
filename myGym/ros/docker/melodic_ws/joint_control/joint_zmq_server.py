#!/usr/bin/env python2
"""
ROS node to send joint trajectories to TIAGo's arm via action client.
Supports left or right arm, defaulting to left. Safe shutdown cancels goals.
Includes test_joint_action to send a random perturbation trajectory.
"""
# export ROS_MASTER_URI=http://tiago-114c:11311
# export ROS_IP=10.68.0.128
from __future__ import print_function
import argparse
import sys
import rospy
import actionlib
from actionlib import CommState, GoalStatus
import threading
import random
import math
import numpy as np
from scipy.interpolate import splprep, splev
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import JointState
from urdf_parser_py.urdf import URDF
import PyKDL as kdl
try:
    from kdl_parser_py.urdf import treeFromParam, treeFromFile, treeFromUrdfModel
except ImportError:
    treeFromParam = None
    from urdf_parser_py.urdf import URDF
    from kdl_parser_py import treeFromUrdfModel, treeFromFile
# from tf_conversions import kdl_parser_py
import zmq
from threading import Thread
from warnings import warn
import cloudpickle as cpl
from zmq.utils.strtypes import asbytes
import tty, termios


def read_char(message=""):
    print(message, end='')
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    ch = ''
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch.lower()


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


def merge_joint_positions(joint_positions, merge_distance=0.01):
    """
    Merge nearby joint-space points into their average.
    - joint_positions: list of lists/ndarrays (N x D)
    - merge_distance: maximum Euclidean distance to merge
    """
    pts = np.array(joint_positions)
    if pts.size == 0:
        return []

    N = len(pts)
    visited = np.zeros(N, bool)
    clusters = []

    # Build pairwise distance matrix
    # Note: moderate N, so O(N^2) ok
    dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)

    for i in range(N):
        if visited[i]:
            continue
        # Find all within threshold (and overlapping clusters)
        mask = (dists[i] < merge_distance)
        indices = set(np.where(mask)[0].tolist())
        # Grow cluster: include neighbors-of-neighbors
        changed = True
        while changed:
            changed = False
            for j in list(indices):
                new = set(np.where(dists[j] < merge_distance)[0])
                if not new.issubset(indices):
                    indices |= new
                    changed = True
        # mark visited & compute centroid
        visited[list(indices)] = True
        cluster_pts = pts[list(indices)]
        centroid = cluster_pts.mean(axis=0)
        clusters.append(centroid.tolist())

    return clusters


def smooth_joint_positions(joint_positions, smoothing_factor=2.0):
    """
    Smooth a trajectory defined by joint_positions: list of N points (each D-dimensional).
    smoothing_factor: user-friendly 1..10 scale (1=almost no smoothing; 10=max smoothing).
    Returns a new N x D smoothed trajectory.
    """
    pts = np.asarray(joint_positions)
    if pts.ndim != 2:
        raise ValueError("Expect shape (N, D)")
    N, D = pts.shape
    # parameterize by arc-length
    u = np.linspace(0, 1, N)

    # Map user scale (1..10) to s parameter
    # Typical s range is (m - sqrt(2m)) ... m + sqrt(2m) where m=N (see docs)
    m = float(N)
    s_min = m - np.sqrt(2*m)
    s_max = m + np.sqrt(2*m)
    # linear interpolate scale to [s_min, s_max]
    sf = float(smoothing_factor)
    sf = max(1.0, min(10.0, sf))
    s = s_min + (s_max - s_min) * ((sf - 1.0) / 9.0)

    # splprep expects list of coordinate arrays
    coord_list = [pts[:, j] for j in range(D)]
    tck, u_out = splprep(coord_list, u=u, s=s, k=3)  # k=3 cubic splines recommended

    # Evaluate at original u
    smooth_coords = splev(u, tck)
    # Reconstruct NxD
    smooth = np.vstack(smooth_coords).T.tolist()
    return smooth


class TiagoTrajectoryController(object):

    def __init__(self,
                 arm_side='right',
                 unsafe=False,
                 smooth=False,
                 merge=0.0,
                 duration=2.0,
                 max_length=0,
                 start_offset=0,
                 execution_delay=3.0,
                 urdf_path=None
                 ):
        # Initialize ROS node
        rospy.init_node('arm_trajectory_controller', anonymous=True)

        self.arm_side = arm_side

        self.goal = None
        self.goal_handle = None

        self.smooth = smooth
        self.merge = merge
        self.duration = duration
        self.max_length = max_length
        self.start_offset = start_offset
        self.execution_delay = execution_delay

        # Define joint names for TIAGo arm
        # TIAGo arms have 7 joints: arm_left_1_joint ... arm_left_7_joint
        self.joint_names = ['arm_{}_{}_joint'.format(self.arm_side, i) for i in range(1, 8)]

        # Storage for current joint states
        self.joint_state = None
        rospy.Subscriber('/joint_states', JointState, self.joint_state_cb)

        # Ensure safe shutdown: cancel goal on exit
        rospy.on_shutdown(self.shutdown_hook)

        # Forward kinematics setup
        if urdf_path is None:
            if treeFromParam is not None:  # Melodic
                ok, tree = treeFromParam('/robot_description')
            else:  # Noetic
                try:
                    robot = URDF.from_parameter_server()
                except BaseException as e:
                    rospy.logerr("Failed to retrieve URDF from parameter server: %s" % e)
                    rospy.signal_shutdown('URDF parse failed')
                ok, tree = treeFromUrdfModel(robot)
        else:
            robot = URDF.from_xml_file(urdf_path)
            ok, tree = treeFromUrdfModel(robot)

        if not ok:
            rospy.logerr("Failed to parse URDF to KDL tree")
            rospy.signal_shutdown('URDF parse failed')
        # base = 'arm_{0}_1_link'.format(self.arm_side)
        self.base = 'torso_lift_link'  # one link before the arm
        self.end = 'arm_{0}_7_link'.format(self.arm_side)
        # print(tree.getNrOfJoints())
        # print(str(tree))
        self.chain = tree.getChain(self.base, self.end)
        # print(self.chain.getSegment(0))
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)

        # Action client for follow_joint_trajectory
        action_server = '/{}arm_{}_controller/follow_joint_trajectory'.format('safe_' if not unsafe else '', self.arm_side)
        self.client = actionlib.ActionClient(action_server, FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for action server: {}".format(action_server))
        if not self.client.wait_for_server(rospy.Duration(10.0)):
            rospy.logerr("Action server not available")
            rospy.signal_shutdown('No action server')
        rospy.loginfo("Connected to {}".format(action_server))

        self.disp_pub = rospy.Publisher('/move_group/display_planned_path', DisplayTrajectory, queue_size=1)
        self.marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=1)

        self.zmq_service_server = ServiceServer(self._data_callback)

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

        # print(joint_names)

        self.send_joint_action(joint_names, joint_positions, self.duration)
        return {'code': 200, 'message': 'success'}

    def joint_state_cb(self, msg):
        # Store the latest joint state
        self.joint_state = msg

    def shutdown_hook(self):
        rospy.loginfo("Shutting down: cancelling any active goal...")
        if self.goal_handle is not None:
            self.goal_handle.cancel()
        self.client.cancel_all_goals()
        # Wait briefly for cancellation confirmation
        # finished_before_timeout = self.client.wait_for_result(rospy.Duration(5.0))
        # if finished_before_timeout:
        #     state = self.client.get_state()
        #     rospy.loginfo("Goal cancelled, final state: %d" % state)
        # else:
        #     rospy.logwarn("Cancellation not confirmed within timeout")

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

    def transition_cb(self, goal_handle):
        comm_state = goal_handle.get_comm_state()
        rospy.loginfo("State transition to %d : %s" % comm_state, CommState.to_string(comm_state))
        # goal_state = goal_handle.get_goal_state()
        # rospy.loginfo("Goal state is %d : %s" % (goal_state, GoalStatus.to_string(goal_state)))

    def feedback_cb(self, goal_handle, feedback):
        # rospy.loginfo_throttle(1.0, "Feedback received: time_from_start=%.2f" % (
        #     feedback.actual.time_from_start.to_sec()))
        rospy.loginfo_throttle(1.0, "Feedback received:\n{}\n{}".format(feedback.actual.positions, feedback.actual.velocities))

    def done_cb(self, state, result):
        rospy.loginfo("Goal done, state\nstate: {}\nresult: {}".format(state, result))

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
        # traj.header.stamp = rospy.Time.now()
        # traj.header.frame_id = "base_link"
        traj.header.frame_id = self.base
        traj.joint_names = joint_names
        points = []

        if self.smooth:
            positions = smooth_joint_positions(positions)
        if self.merge > 0.0:
            positions = merge_joint_positions(positions, self.merge)

        if self.start_offset > 0:
            if self.start_offset > len(positions):
                rospy.logerr("start_offset too large: %d (offset) > %d (trajectory length)" % (self.start_offset, len(positions)))
                return
            positions = positions[self.start_offset:]
        if self.max_length > 0:
            positions = positions[:self.max_length]

        for pos, dur in zip(positions, durations):
            pt = JointTrajectoryPoint()
            pt.positions = pos
            pt.time_from_start = rospy.Duration(dur)
            points.append(pt)

        traj.points = points

        self.visualize_trajectory_fk(traj)
        key = read_char("Execute goal? (y/n): ")
        while key not in ['y', 'n']:
            key = read_char("Wrong input {}. Execute goal? (y/n): ".format(key))
        if key == 'n':
            rospy.logwarn("Aborting goal.")
            return

        self.goal = FollowJointTrajectoryGoal()
        self.goal.trajectory = traj
        self.goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(self.execution_delay)
        rospy.loginfo("Sending test goal:")
        rospy.loginfo(str(self.goal))

        # Send goal
        self.goal_handle = self.client.send_goal(self.goal, feedback_cb=self.feedback_cb)
        rospy.loginfo("Goal sent, waiting for result...")
        # self.client.wait_for_result(rospy.Duration(100.0))
        # state = self.client.get_state()
        # rospy.loginfo("Action completed with state: %d" % state)
        # self.goal = None

    # def send_goal_threaded(self, traj):
    #     traj.header.stamp = rospy.Time(0)
    #     goal = FollowJointTrajectoryGoal(trajectory=traj)

    #     def send_n_wait():
    #         self.client.send_goal(goal, done_cb=self._done_cb, feedback_cb=self._feedback_cb)
    #         self.client.wait_for_result()
    #         state = self.client.get_state()
    #         rospy.loginfo("Action completed with state: %d" % state)

    #     goal_thread = threading.Thread(target=send_n_wait)
    #     goal_thread.start()

    def test_joint_sending(self, num_points, max_deg):
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
            points.append(new_positions)
            prev_positions = new_positions

        self.send_joint_action(traj.joint_names, points)

    # def test_joint_action(self, num_points, max_deg):
    #     """
    #     Send a test trajectory of num_points, where each joint is perturbed
    #     by at most max_deg degrees from previous point.
    #     """
    #     rospy.loginfo("Generating test trajectory: %d points, max %.2f deg" % (num_points, max_deg))
    #     # Get current joint positions
    #     init_positions = self.get_current_positions()
    #     if init_positions is None:
    #         return
    #     max_rad = math.radians(max_deg)

    #     # Build trajectory
    #     traj = JointTrajectory()
    #     traj.joint_names = self.joint_names
    #     points = []
    #     prev_positions = init_positions

    #     for i in range(num_points):
    #         new_positions = []
    #         for p in prev_positions:
    #             new_positions.append(p + random.uniform(-max_rad, max_rad))
    #         pt = JointTrajectoryPoint()
    #         pt.positions = new_positions
    #         # simple timing: 1 second apart
    #         pt.time_from_start = rospy.Duration((i + 1) * 1.0)
    #         points.append(pt)
    #         prev_positions = new_positions

    #     traj.points = points
    #     self.goal = FollowJointTrajectoryGoal()
    #     self.goal.trajectory = traj
    #     # Stamp a little in the future
    #     # goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(0.1)
    #     # goal.trajectory.header.stamp = rospy.Time.now()
    #     self.goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(5.0)

    #     rospy.loginfo("Sending test goal:")
    #     rospy.loginfo(str(self.goal))

    #     # Send goal
    #     self.client.send_goal(self.goal, feedback_cb=self.feedback_cb)
    #     rospy.loginfo("Test goal sent, waiting for result...")
    #     self.client.wait_for_result(rospy.Duration(30.0))
    #     state = self.client.get_state()
    #     rospy.loginfo("Test action completed with state: %d" % state)
    #     self.goal = None

    def visualize_trajectory_fk(self, traj):
        robot_traj = RobotTrajectory(joint_trajectory=traj)
        disp = DisplayTrajectory(
            trajectory=[robot_traj],
        )
        disp.trajectory_start.joint_state = self.joint_state

        self.disp_pub.publish(disp)

        marker_array = MarkerArray()
        line = Marker()
        line.header.frame_id = self.base
        line.header.stamp = rospy.Time.now()
        line.ns = 'ee_path'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.005  # Line width
        line.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)
        line.points = []

        marker_array.markers.append(line)

        print("Segments:" + str(self.chain.getNrOfSegments()))
        print("Joints:" + str(self.chain.getNrOfJoints()))

        for i in range(self.chain.getNrOfSegments()):
            print("Segment {}: {}".format(i, self.chain.getSegment(i).getName()))

        # Additionally compute end-effector path:
        path = []
        for idx, pt in enumerate(traj.points):
            jarr = kdl.JntArray(len(pt.positions))
            for i, q in enumerate(pt.positions):
                jarr[i] = q
            tcp_pose = kdl.Frame()
            self.fk_solver.JntToCart(jarr, tcp_pose)
            pose = tcp_pose.p
            p = Point(x=pose.x(), y=pose.y(), z=pose.z())
            path.append((pose.x(), pose.y(), pose.z()))
            # import pdb; pdb.set_trace()

            print("Position:", tcp_pose.p)
            print("Orientation (rotation matrix):", tcp_pose.M)

            # Add point to line
            marker_array.markers[0].points.append(p)

            # Add sphere marker at point
            sphere = Marker()
            sphere.header = marker_array.markers[0].header
            sphere.ns = 'ee_points'
            sphere.id = idx + 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = p
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.02
            sphere.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
            marker_array.markers.append(sphere)

        self.marker_pub.publish(marker_array)
        rospy.loginfo("Computed FK positions: %s", path)


def main():
    parser = argparse.ArgumentParser(description='TIAGo arm trajectory controller')
    parser.add_argument('--side', choices=['left', 'right'], default='left',
                        help='Arm side to control (left or right)')
    parser.add_argument('--test', nargs=2, type=float, metavar=('NUM_POINTS', 'MAX_DEG'),
                        help='Run test_joint_action with NUM_POINTS and MAX_DEG degrees')
    parser.add_argument('--unsafe', action='store_true', help='Using the "unsafe" action client')
    parser.add_argument('--max-length', '--ml', type=int, default=0, help='Maximum length of the trajectory. Default: 0 (no limit)')
    parser.add_argument('--start-offset', '-o', type=int, default=0, help='Start offset of the trajectory. Default: 0 (no offset)')
    parser.add_argument('--smooth', '-s', action='store_true', help='Smooth the trajectory')
    parser.add_argument('--merge', '-m', type=float, default=0.0, help='Merge trajectory points closer than this value (in meters)')
    parser.add_argument('--duration', '-d', type=float, default=2.0, help='Duration of each trajectory point (in seconds)')
    parser.add_argument( '--urdf-path', '--urdf', '-u', type=str, default=None, help='URDF path to load from, instead of using ROS param.')
    args = parser.parse_args()

    arm_side = args.side
    unsafe = args.unsafe
    smooth = args.smooth
    if not np.isfinite(args.merge) or args.merge < 0.0:
        merge = 0.0
    else:
        merge = args.merge
    if not np.isfinite(args.duration) or args.duration < 0.1:
        duration = 2.0
    else:
        duration = args.duration
    if not np.isfinite(args.max_length) or args.max_length < 0:
        max_length = 0
    else:
        max_length = int(args.max_length)
    if not np.isfinite(args.start_offset) or args.start_offset < 0:
        start_offset = 0
    else:
        start_offset = int(args.start_offset)
    controller = TiagoTrajectoryController(
        arm_side=arm_side,
        unsafe=unsafe,
        smooth=smooth,
        merge=merge,
        duration=duration,
        max_length=max_length,
        start_offset=start_offset,
        urdf_path=args.urdf_path
    )

    if args.test:
        num_points, max_deg = int(args.test[0]), args.test[1]
        rospy.sleep(1.0)  # wait for subscribers
        # controller.test_joint_action(num_points, max_deg)
        controller.test_joint_sending(num_points, max_deg)
    else:
        rospy.loginfo("Ready to send trajectories. Use controller.test_joint_action() in code or extend this script.")
        rospy.spin()


if __name__ == '__main__':
    main()

# import rospy
# import os
# import argparse

# def main(dump_path):
#     rospy.init_node("dump_urdf")
#     urdf_str = rospy.get_param("/robot_description")
#     with open(os.path.join(dump_path, "robot.urdf"), "w") as f:
#         f.write(urdf_str)

# if __name__ == "__main__":
#     argparse = argparse.ArgumentParser()
#     argparse.add_argument("dump_path", type=str, default="/tmp")
#     args = argparse.parse_args()
#     dump_path = args.dump_path
#     main(dump_path=dump_path)

# for i, q in enumerate(np.random.rand(7)): jarr[i] = q
