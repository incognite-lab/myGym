#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
TIAGo trajectory bridge (Python 2.7, ROS Melodic)

- Receives joint trajectories over ZeroMQ (cloudpickle or JSON)
- Trimming controls: --start-offset/-o, --trim/-t, --max-length/--ml/-l
- Merges close points, smooths optionally, and re-times by joint-space distance
- Prints trajectory and REQUIRES 'y' confirmation before sending
- Sends to FollowJointTrajectory action (safe/unsafe arm controllers)
- Robust preemption via SimpleActionClient + stop_tracking_goal()
- RViz preview (DisplayTrajectory + Markers)
- Approximate self-collision check using URDF + KDL + bounding spheres
"""

from __future__ import print_function, division
import enum
import sys, os, argparse, json, threading, math, time
import numpy as np

import rospy
import actionlib
from actionlib import GoalStatus
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import JointState

from urdf_parser_py.urdf import URDF
import PyKDL as kdl
try:
    from kdl_parser_py.urdf import treeFromParam, treeFromUrdfModel
except Exception:
    treeFromParam = None
    from kdl_parser_py import treeFromUrdfModel

from enum import Enum

# SciPy optional; fallback to Catmull-Rom if absent
try:
    from scipy.interpolate import splprep, splev
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ZeroMQ
import zmq
import cloudpickle as cpl

# -------- TTY confirmation --------
import tty, termios


class KdlError(Enum):
    E_DEGRADED = 1
    E_NOERROR = 0
    E_NO_CONVERGE = -1
    E_UNDEFINED = -2
    E_NOT_UP_TO_DATE = -3
    E_SIZE_MISMATCH = -4
    E_MAX_ITERATIONS_EXCEEDED = -5
    E_OUT_OF_RANGE = -6
    E_NOT_IMPLEMENTED = -7
    E_SVD_FAILED = -8


def read_char(message=""):
    sys.stdout.write(message)
    sys.stdout.flush()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch.lower()

# ---------- math & utils ----------

def joint_distance(p, q):
    """Euclidean distance in joint space (assume radians)."""
    p = np.asarray(p); q = np.asarray(q)
    return float(np.linalg.norm(q - p, ord=2))

def merge_consecutive(points, eps):
    """Greedy *sequential* merge of consecutive points closer than eps (radians)."""
    if not points: return []
    out = [np.asarray(points[0], dtype=float)]
    run_sum = out[0].copy()
    run_n = 1
    for i in range(1, len(points)):
        cur = np.asarray(points[i], dtype=float)
        if joint_distance(out[-1], cur) < eps:
            run_sum += cur; run_n += 1
            out[-1] = run_sum / float(run_n)
        else:
            run_sum = cur.copy(); run_n = 1
            out.append(cur)
    return [p.tolist() for p in out]

def catmull_rom(points, upsample=1):
    """Lightweight C^1 smoothing without SciPy. Densifies by upsample per segment."""
    P = [np.asarray(p, dtype=float) for p in points]
    if len(P) <= 2 or upsample <= 1:
        return [p.tolist() for p in P]
    Q = []
    N = len(P)
    for i in range(N - 1):  # real segments only
        p0 = P[i-1] if i > 0 else P[i]
        p1 = P[i]
        p2 = P[i+1]
        p3 = P[i+2] if i+2 < N else P[i+1]
        for s in range(int(upsample)):  # t in [0,1)
            t  = float(s) / float(upsample)
            t2 = t * t; t3 = t2 * t
            q = 0.5 * ((2.0*p1) + (-p0+p2)*t + (2.0*p0-5.0*p1+4.0*p2-p3)*t2 + (-p0+3.0*p1-3.0*p2+p3)*t3)
            if i == 0 and s == 0 or s > 0:  # keep first, then all s>0
                Q.append(q)
    Q.append(P[-1])
    return [q.tolist() for q in Q]


def smooth_points(points, smoothing=0.0, resample=None):
    """
    Smooth using cubic splines if SciPy is available, else Catmull-Rom.
    smoothing: 0..1 (0=interpolate, 1=heavier smoothing)
    resample: None (keep length) or integer total points to sample.
    """
    P = np.asarray(points, dtype=float)
    N, D = P.shape
    if N <= 3:
        return [p.tolist() for p in P]
    # arc-length parameterization
    arc = [0.0]
    for i in range(1, N):
        arc.append(arc[-1] + joint_distance(P[i-1], P[i]))
    arc = np.asarray(arc, dtype=float)
    if arc[-1] == 0.0:
        return [p.tolist() for p in P]
    u = arc / arc[-1]

    if resample is None: resample = N
    u_new = np.linspace(0., 1., int(resample))

    if SCIPY_OK:
        coord = [P[:, j] for j in range(D)]
        # map smoothing 0..1 to s parameter ~ [0, N]
        s = float(smoothing) * float(N)  # map 0..1 -> [0..N]
        tck, _ = splprep(coord, u=u, s=s, k=3)
        sm = splev(u_new, tck)
        Q = np.vstack(sm).T
        return [q.tolist() for q in Q]
    else:
        # lightweight fallback when no scipy
        up = int(max(1, round(float(resample) / float(max(1, N-1)))))
        Q = catmull_rom(P, upsample=up)
        if len(Q) != resample:
            # simple uniform resample to exactly resample points
            idx = np.linspace(0, len(Q)-1, resample)
            QQ = []
            for t in idx:
                i = int(math.floor(t))
                a = t - i
                a = max(0.0, min(1.0, a))
                A = np.asarray(Q[i]); B = np.asarray(Q[min(i+1, len(Q)-1)])
                QQ.append((1.0-a)*A + a*B)
            Q = [q.tolist() for q in QQ]
        return Q

def retime_by_distance(points, v_max=0.7, a_max=1.5, dt_min=0.08, t0=0.08):
    """
    Compute cumulative times (seconds) for each point based on joint-space distance.
    dt_i = max(dist/v_max, sqrt(2*dist/a_max), dt_min)
    Returns list of monotonically increasing times (same length as points).
    """
    times = [t0]
    for i in range(1, len(points)):
        dist = joint_distance(points[i-1], points[i])
        dt_v = dist / max(1e-6, float(v_max))
        dt_a = math.sqrt(2.0 * dist / max(1e-6, float(a_max)))
        dt = max(dt_min, dt_v, dt_a)
        times.append(times[-1] + dt)
    return times

# ---------- trimming helpers ----------

def apply_start_offset(pts, offset):
    if offset <= 0: return list(pts)
    if offset >= len(pts): return []
    return list(pts[offset:])

def apply_trim_end(pts, trim):
    if trim <= 0: return list(pts)
    if trim >= len(pts): return []
    return list(pts[:len(pts)-trim])

def apply_max_length(pts, max_len):
    if max_len is None or max_len <= 0: return list(pts)
    return list(pts[:max_len])

# ---------- ZeroMQ service ----------

class ZmqServer(object):
    """Simple REP server supporting cloudpickle or JSON payloads."""
    def __init__(self, callback, port=242424, addr="0.0.0.0", protocol="tcp"):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.SNDTIMEO, 3000)
        self._sock.setsockopt(zmq.RCVTIMEO, 3000)
        self._addr = "%s://%s:%d" % (protocol, addr, port)
        print("ZMQ REP listening on %s" % self._addr)
        self._sock.bind(self._addr)
        self._cb = callback
        self._alive = True
        self._th = threading.Thread(target=self._loop)
        self._th.daemon = True
        self._th.start()

    def _loop(self):
        while self._alive and not rospy.is_shutdown():
            try:
                req = self._sock.recv()
            except zmq.Again:
                continue
            # Try cloudpickle first, then JSON
            try:
                data = cpl.loads(req)
            except Exception:
                try:
                    data = json.loads(req.decode('utf-8'))
                except Exception as e:
                    data = {"cmd": "invalid", "error": str(e)}
            try:
                resp = self._cb(data) or {}
            except Exception as e:
                rospy.logerr("ZMQ callback error: %s", e)
                resp = {"ok": False, "error": str(e)}
            try:
                self._sock.send(cpl.dumps(resp, protocol=2))
            except zmq.Again:
                rospy.logwarn("ZMQ send timeout, resetting socket")
                self._sock.close()
                self._sock = self._ctx.socket(zmq.REP)
                self._sock.bind(self._addr)

    def close(self):
        self._alive = False
        self._th.join(1.0)
        try:
            self._sock.close()
        except Exception:
            pass


# ---------- Main controller ----------
class TiagoTrajectoryController(object):
    def __init__(self, arm_side='left', unsafe=False,
                 merge_eps=0.0, smooth=0.0, resample=0,
                 v_max=0.7, a_max=1.5, dt_min=0.08,
                 exec_delay=0.2, urdf_path=None, zmq_port=242424,
                 start_offset=0, trim_end=0, max_length=0,
                 no_wait_server=False,
                 ):
        """
        merge_eps: radians threshold for sequential merging
        smooth: 0..1 smoothing weight; resample: 0 keep length, else count
        v_max, a_max, dt_min: timing params
        exec_delay: seconds to delay header.stamp
        start_offset: drop first K input poses
        trim_end: drop last T input poses
        max_length: cap final number of poses AFTER processing (0 = unlimited)
        """
        self.arm_side = arm_side
        self.safe = (not unsafe)
        self.merge_eps = max(0.0, float(merge_eps))
        self.smooth_w = max(0.0, min(1.0, float(smooth)))
        self.resample = int(resample) if resample and resample > 0 else 0
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.dt_min = float(dt_min)
        self.exec_delay = float(exec_delay)
        self.start_offset = max(0, int(start_offset))
        self.trim_end = max(0, int(trim_end))
        self.max_length = max(0, int(max_length))

        rospy.init_node('tiago_traj_bridge', anonymous=True)

        # Controller names (safe/unsafe)
        base_ns = ('/safe_' if self.safe else '/') + 'arm_%s_controller' % self.arm_side
        self.action_name = base_ns + '/follow_joint_trajectory'
        self.topic_cmd = base_ns + '/command'

        # Action client
        self.client = actionlib.SimpleActionClient(self.action_name, FollowJointTrajectoryAction)
        if no_wait_server:
            rospy.loginfo("Not waiting for action server: %s", self.action_name)
        else:
            rospy.loginfo("Waiting for action server: %s", self.action_name)
            if not self.client.wait_for_server(rospy.Duration(15)):
                raise RuntimeError("No action server: %s" % self.action_name)
        ## Gripper controller ----------------------------------------------------------------------------------
        gripper_base_ns = 'gripper_%s_controller' % self.arm_side
        self.gripper_action_name = gripper_base_ns + '/follow_joint_trajectory'
        
        self.gripper_client = actionlib.SimpleActionClient(
            self.gripper_action_name,
            FollowJointTrajectoryAction,
        )
        if no_wait_server:
            rospy.loginfo("Not waiting for GRIPPER action server:%s", self.gripper_action_name)
        else:
            rospy.loginfo("Waiting for GRIPPER action server:%s", self.gripper_action_name)
            try:
                self.gripper_client.wait_for_server(rospy.Duration(5.0))
            except rospy.ROSException:
                rospy.logwarn("No gripper action server at %s (gripper will nout move)", self.gripper_action_name)
                self.gripper_client = None
                
        ##-------------------------------------------------------------------------------------------------------

        # Publishers for visualization
        self.disp_pub = rospy.Publisher('/move_group/display_planned_path', DisplayTrajectory, queue_size=1)
        self.marker_pub = rospy.Publisher('/tiago_traj_preview', MarkerArray, queue_size=1)
        self.cmd_pub = rospy.Publisher(self.topic_cmd, JointTrajectory, queue_size=1)

        # Joint state buffer
        self.joint_state_lock = threading.Lock()
        self.joint_state = None
        rospy.Subscriber('/joint_states', JointState, self._joint_state_cb)

        # URDF/KDL for FK and simple self-collision checks
        if urdf_path is None:
            if treeFromParam is not None:
                ok, self.tree = treeFromParam('/robot_description')
                if not ok:
                    raise RuntimeError("Failed to parse URDF from /robot_description")
                robot = None
            else:
                robot = URDF.from_parameter_server()
                ok, self.tree = treeFromUrdfModel(robot)
                if not ok:
                    raise RuntimeError("Failed to build KDL tree")
        else:
            robot = URDF.from_xml_file(urdf_path)
            ok, self.tree = treeFromUrdfModel(robot)
            if not ok:
                raise RuntimeError("Failed to build KDL tree from file")

        self.base_link = 'torso_fixed_link'
        # self.ee_link = 'arm_%s_7_link' % self.arm_side
        self.ee_link = 'gripper_%s_link' % self.arm_side
        self.chain = self.tree.getChain(self.base_link, self.ee_link)
        self.tree.getChain('torso_fixed_link', 'arm_right_7_link').getNrOfJoints()
        self.fk = kdl.ChainFkSolverPos_recursive(self.chain)

        # arm link names for approx collision
        self.arm_links = []
        for i in range(self.chain.getNrOfSegments()):
            self.arm_links.append(self.chain.getSegment(i).getName())
        self.bounds = self._build_bounding_spheres(robot)

        # ZMQ
        self.server = ZmqServer(self._on_request, port=zmq_port)
        rospy.on_shutdown(self._on_shutdown)

        # goal tracking / synchronization
        self._goal_lock = threading.Lock()
        self._goal_active = False
        self._done_event = threading.Event()

        rospy.loginfo("Ready. Controller: %s", self.action_name)

    # ---- ROS plumbing ----
    def _joint_state_cb(self, msg):
        with self.joint_state_lock:
            self.joint_state = msg

    def _current_positions(self, names):
        deadline = rospy.Time.now() + rospy.Duration(5.0)
        while not rospy.is_shutdown():
            with self.joint_state_lock:
                js = self.joint_state
            if js is not None:
                name2pos = dict(zip(js.name, js.position))
                try:
                    return [float(name2pos[n]) for n in names]
                except KeyError:
                    pass
            if rospy.Time.now() > deadline:
                return None
            rospy.sleep(0.02)

    # ---- ZMQ handler ----
    def _on_request(self, data):
        """
        Accepts:
         - {"cmd":"send","joint_names":[...],"data":[[...],...], optional trimming/merge/smooth/timing keys}
         - {"cmd":"cancel"}
         - {"cmd":"halt"}  # cancel + hold at current pos
        Backward compatible with old dicts having 'joint_names' and 'data'.
        """
        try:
            cmd = data.get('cmd', 'send')
        except Exception:
            cmd = 'send'

        if cmd == 'send':
            joint_names = data.get('joint_names', None)
            positions = data.get('data', None)
            # DebugLog
            rospy.loginfo("ZMQ recv: %d joints, %d points", len(joint_names), len(positions))
            if positions:
                rospy.loginfo("first point len = %d", len(positions[0]))
            rospy.loginfo("joint_name = %s", joint_names)
            
            if not positions or not joint_names:
                return {"ok": False, "err": "empty trajectory or names"}

            # normalize names
            names = []
            for nm in joint_names:
                s = str(nm).replace('_gjoint', '_joint').replace('_rjoint', '_joint').replace("b'", '').replace("'", '')
                names.append(s)

            # per-request overrides
            merge = float(data.get('merge', self.merge_eps))
            smooth = float(data.get('smooth', self.smooth_w))
            resample = int(data.get('resample', self.resample))
            v_max = float(data.get('v_max', self.v_max))
            a_max = float(data.get('a_max', self.a_max))
            dt_min = float(data.get('dt_min', self.dt_min))
            preview_only = bool(data.get('preview_only', False))

            # trimming controls (per-request or defaults)
            start_offset = int(data.get('start_offset', self.start_offset))
            trim_end = int(data.get('trim', self.trim_end))
            max_length = int(data.get('max_length', self.max_length))

            ok, warn = self.send_trajectory(
                names, positions,
                merge=merge, smooth=smooth, resample=resample,
                v_max=v_max, a_max=a_max, dt_min=dt_min,
                start_offset=start_offset, trim_end=trim_end, max_length=max_length,
                preview_only=preview_only
            )
            return {"ok": ok, "warning": warn}

        elif cmd == 'cancel':
            self.cancel_goal()
            return {"ok": True}
        elif cmd == 'halt':
            self.halt()
            return {"ok": True}
        else:
            return {"ok": False, "err": "unknown cmd: %r" % cmd}

    # ---- collision helpers ----
    def _build_bounding_spheres(self, robot):
        bounds = {}
        default_r = 0.06
        for ln in self.arm_links:
            bounds[ln] = ((0.0, 0.0, 0.0), default_r)
        try:
            if robot is None: return bounds
            for ln in self.arm_links:
                if ln not in robot.link_map: continue
                L = robot.link_map[ln]
                if L.collision and L.collision.geometry:
                    g = L.collision.geometry
                    if hasattr(g, 'radius') and g.radius:
                        r = float(g.radius)
                    elif hasattr(g, 'size') and g.size:
                        sx, sy, sz = g.size
                        r = 0.5 * math.sqrt(sx*sx + sy*sy + sz*sz)
                    elif hasattr(g, 'length') and hasattr(g, 'radius'):
                        r = float(g.radius) + 0.5*float(getattr(g, 'length', 0.0))
                    else:
                        r = default_r
                    off = (0.0, 0.0, 0.0)
                    if L.collision.origin and L.collision.origin.xyz:
                        off = tuple(L.collision.origin.xyz)
                    bounds[ln] = (off, max(default_r, r))
        except Exception as e:
            rospy.logwarn("URDF sphere extraction failed, using defaults: %s", e)
        return bounds

    def check_self_collision(self, joint_names, points):
        if len(points) == 0: return []
        pairs = []
        indices = []
        for i, ln_i in enumerate(self.arm_links):
            if ln_i not in self.bounds: continue
            for j, ln_j in enumerate(self.arm_links):
                if j <= i + 1: continue
                if ln_j not in self.bounds: continue
                pairs.append((ln_i, ln_j)); indices.append((i, j))
        result = []
        jnt_size = len(points[0])
        for k, q in enumerate(points):
            jarr = kdl.JntArray(jnt_size)
            for i, val in enumerate(q): jarr[i] = float(val)
            frames = []
            f = kdl.Frame()
            for seg_idx in range(self.chain.getNrOfSegments()):
                self.fk.JntToCart(jarr, f, seg_idx+1)
                frames.append(kdl.Frame(f))
            any_collision = False
            for (ln_i, ln_j), (i, j) in zip(pairs, indices):
                c_i, r_i = self.bounds.get(ln_i, ((0,0,0), 0.0))
                c_j, r_j = self.bounds.get(ln_j, ((0,0,0), 0.0))
                p_i = frames[i].p + frames[i].M * kdl.Vector(c_i[0], c_i[1], c_i[2])
                p_j = frames[j].p + frames[j].M * kdl.Vector(c_j[0], c_j[1], c_j[2])
                dx = p_i.x()-p_j.x(); dy = p_i.y()-p_j.y(); dz = p_i.z()-p_j.z()
                if dx*dx + dy*dy + dz*dz < (r_i + r_j - 0.01)**2:
                    any_collision = True; break
            result.append(any_collision)
        return result

    # ---- visualization ----
    def visualize(self, traj):
        # DisplayTrajectory (no MoveIt requirement)
        # disp = DisplayTrajectory()
        # disp.trajectory.append(RobotTrajectory(joint_trajectory=traj))
        # with self.joint_state_lock:
        #     if self.joint_state is not None:
        #         disp.trajectory_start.joint_state = self.joint_state
        # self.disp_pub.publish(disp)

        # EE path & collision markers
        marr = MarkerArray()
        line = Marker()
        line.header.frame_id = self.base_link
        line.header.stamp = rospy.Time.now()
        line.ns = 'ee_path'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.003
        line.pose.orientation.w = 1.0
        line.color = ColorRGBA(0.1, 0.9, 0.1, 1.0)
        marr.markers.append(line)

        for k, pt in enumerate(traj):
            # fk to ee (quick demo path)
            jarr = kdl.JntArray(len(pt))
            for i, q in enumerate(pt):
                jarr[i] = q
            frame = kdl.Frame()
            err_code = self.fk.JntToCart(jarr, frame)
            if err_code < 0:
                rospy.logwarn("Vizualization: FK error: %d '%s' for point %d", err_code, KdlError(err_code).name, k)
            p = frame.p
            # import pdb; pdb.set_trace()
            gp = Point(p.x(), p.y(), p.z())
            marr.markers[0].points.append(gp)

            m = Marker()
            m.header = line.header
            m.ns = 'ee_pos'; m.id = k + 1
            m.type = Marker.SPHERE; m.action = Marker.ADD
            m.pose.position = gp; m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.006
            m.color = ColorRGBA(1.0, 0.2, 0.2, 0.9)
            marr.markers.append(m)
        self.marker_pub.publish(marr)

    # ---- printing and confirmation ----
    def _print_trajectory(self, traj):
        print("\n=== Trajectory Preview ===")
        print("frame_id  : %s" % (traj.header.frame_id or ''))
        print("stamp     : now + %.3f s" % self.exec_delay)
        print("joint_names (%d): %s" % (len(traj.joint_names), ', '.join(traj.joint_names)))
        print("points (%d):" % len(traj.points))
        for i, pt in enumerate(traj.points):
            t = pt.time_from_start.to_sec()
            pos_str = ", ".join(["% .5f" % x for x in pt.positions])
            print("  #%03d  t=%.4f : [%s]" % (i, t, pos_str))
        print("==========================\n")

    def _confirm_send(self):
        key = read_char("Execute goal? (y/n): ")
        while key not in ('y', 'n'):
            key = read_char("Please press 'y' or 'n' (you pressed '%s'): " % key)
        return key == 'y'

    # ---- building and sending ----
    def build_traj(self, joint_names, points, times):
        # separate torso, arm and gripper joints (if any)
        joint_names_arm = []
        joint_names_torso = []
        joint_names_gripper = []
        jcodes = np.zeros(len(joint_names))
        for i, jname in enumerate(joint_names):
            if jname.startswith('arm_'):
                joint_names_arm.append(jname)
                jcodes[i] = 0
            elif jname.startswith('torso_'):
                joint_names_torso.append(jname)
                jcodes[i] = 1
            elif jname.startswith('gripper_'):
                joint_names_gripper.append(jname)
                jcodes[i] = 2
            else:
                raise ValueError("Unknown joint type '%s'" % jname)

        jt_arm = JointTrajectory()
        jt_arm.header.stamp = rospy.Time.now() + rospy.Duration(self.exec_delay)
        jt_arm.header.frame_id = self.base_link
        jt_arm.joint_names = list(joint_names_arm)
        jt_torso = JointTrajectory()
        jt_torso.header.stamp = rospy.Time.now() + rospy.Duration(self.exec_delay)
        jt_torso.header.frame_id = self.base_link
        jt_torso.joint_names = list(joint_names_torso)
        jt_gripper = JointTrajectory()
        jt_gripper.header.stamp = rospy.Time.now() + rospy.Duration(self.exec_delay)
        jt_gripper.header.frame_id = self.base_link
        jt_gripper.joint_names = list(joint_names_gripper)

        joint_trajectories = [jt_arm, jt_torso, jt_gripper]
        for point, t in zip(points, times):
            point = np.array(point)
            for jtype in range(3):
                jt_point = point[jcodes == jtype]
                if len(jt_point) == 0:
                    continue
                pt = JointTrajectoryPoint()
                pt.positions = list(map(float, jt_point))
                pt.time_from_start = rospy.Duration(float(t))
                joint_trajectories[jtype].points.append(pt)
        return jt_arm, jt_torso, jt_gripper

    def send_trajectory(self, joint_names, positions,
                        merge=None, smooth=None, resample=None,
                        v_max=None, a_max=None, dt_min=None,
                        start_offset=0, trim_end=0, max_length=0,
                        preview_only=False):
        # defaults
        if merge is None: merge = self.merge_eps
        if smooth is None: smooth = self.smooth_w
        if resample is None: resample = self.resample
        if v_max is None: v_max = self.v_max
        if a_max is None: a_max = self.a_max
        if dt_min is None: dt_min = self.dt_min

        # --- pipeline ---
        pts = [list(map(float, p)) for p in positions]

        # 1) trimming at input level
        if start_offset > 0:
            pts = apply_start_offset(pts, start_offset)
        if trim_end > 0:
            pts = apply_trim_end(pts, trim_end)
        if len(pts) == 0:
            rospy.logwarn("Trajectory empty after start-offset/trim.")
            return False, "empty_after_trim"

        # 2) merge/smooth/resample
        if merge > 0.0:
            pts = merge_consecutive(pts, merge)
        if smooth > 0.0 or (resample and resample > 0):
            pts = smooth_points(pts, smoothing=smooth, resample=(resample or len(pts)))

        # 3) cap final length (AFTER processing)
        if max_length and max_length > 0:
            pts = apply_max_length(pts, max_length)
        if len(pts) == 0:
            rospy.logwarn("Trajectory empty after max-length trimming.")
            return False, "empty_after_maxlen"

        # 4) timing
        times = retime_by_distance(pts, v_max=v_max, a_max=a_max, dt_min=dt_min, t0=dt_min)

        self.visualize(pts)

        # build trajectory
        arm_traj, torso_traj, gripper_traj = self.build_traj(joint_names, pts, times)

        # approximate collision check
        # collisions = self.check_self_collision(joint_names, pts)
        # n_bad = sum(1 for c in collisions if c)
        # warn = None
        # if n_bad > 0:
        #     warn = "approx_self_collision_suspected_in_%d_points" % n_bad
        #     rospy.logwarn("Approx. self-collision suspected in %d points (arm only).", n_bad)

        # preview (RViz) + print to console
        # self.visualize(traj, collisions)
        self._print_trajectory(arm_traj)
        self._print_trajectory(gripper_traj)

        if preview_only:
            return True, "preview_only"

        # confirmation
        if not self._confirm_send():
            rospy.logwarn("User declined to execute the goal.")
            return True, "user_declined"

        # send goal, with goal-tracking guard
        with self._goal_lock:
            # If a previous goal is still around, cancel & give the server a moment
            if self._goal_active:
                self.client.cancel_goal()
                self._done_event.wait(0.2)  # small grace period
                self.client.stop_tracking_goal()
                self._goal_active = False

            goal = FollowJointTrajectoryGoal()
            goal.trajectory = arm_traj
            self._done_event.clear()
            self._goal_active = True
            self.client.send_goal(goal,
                                  done_cb=self._done_cb,
                                  active_cb=self._active_cb,
                                  feedback_cb=self._feedback_cb)
            rospy.loginfo("Goal sent: %d points (v_max=%.2f, a_max=%.2f, dt_min=%.2f)",
                          len(pts), v_max, a_max, dt_min)
            ## GRIPPER goal-------------------------------------------------------------------------------------
            if self.gripper_client is not None and gripper_traj is not None and len(gripper_traj.points) > 0:
                try:
                    gripper_goal = FollowJointTrajectoryGoal()
                    gripper_goal.trajectory = gripper_traj
                    self.gripper_client.send_goal(gripper_goal)
                    rospy.loginfo("Sent GRIPPER trajectory with %d points.", len(gripper_traj.points))
                except Exception as e:
                    rospy.loginfo("Failed to send gripper trajectory: %s", e)
            ##-------------------------------------------------------------------------------------------------
        return True, "goal_sent"

    # ---- action client callbacks & helpers ----
    def _active_cb(self):
        rospy.loginfo("Goal is ACTIVE.")

    def _feedback_cb(self, fb):
        rospy.logdebug("Feedback t=%.3f", fb.actual.time_from_start.to_sec())

    def _done_cb(self, status, result):
        rospy.loginfo("Goal DONE. Status: %d", status)
        with self._goal_lock:
            self._goal_active = False
            try:
                # Important to ignore late status updates from server
                self.client.stop_tracking_goal()
            except Exception:
                pass
        self._done_event.set()

    def cancel_goal(self):
        with self._goal_lock:
            try: self.client.cancel_goal()
            except Exception: pass
        # give time for PREEMPTING/RECALLED to propagate
        self._done_event.wait(0.15)
        try: self.client.stop_tracking_goal()
        except Exception: pass
        with self._goal_lock:
            self._goal_active = False

    def halt(self, hold_dt=0.15):
        """Cancel and send a one-point 'hold' goal at current position."""
        self.cancel_goal()
        names = ['arm_%s_%d_joint' % (self.arm_side, i) for i in range(1, 8)]
        cur = self._current_positions(names)
        if cur is None:
            rospy.logwarn("Cannot read /joint_states to build hold goal; aborting halt.")
            return
        jt = self.build_traj(names, [cur], [hold_dt])
        goal = FollowJointTrajectoryGoal(trajectory=jt)
        with self._goal_lock:
            self._done_event.clear()
            self._goal_active = True
            self.client.send_goal(goal,
                                  done_cb=self._done_cb,
                                  active_cb=self._active_cb,
                                  feedback_cb=self._feedback_cb)
        rospy.loginfo("Hold goal sent at current joint positions.")

    def _on_shutdown(self):
        try: self.cancel_goal()
        except Exception: pass
        try: self.server.close()
        except Exception: pass


# ---- entry point ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--side', choices=['left', 'right'], default='right')
    parser.add_argument('--unsafe', action='store_true')

    # merge/smooth/timing
    parser.add_argument('--merge', type=float, default=0.0)
    parser.add_argument('--smooth', type=float, default=0.0)
    parser.add_argument('--resample', type=int, default=0)
    parser.add_argument('--v-max', type=float, default=0.7)
    parser.add_argument('--a-max', type=float, default=1.5)
    parser.add_argument('--dt-min', type=float, default=0.08)
    parser.add_argument('--exec-delay', type=float, default=0.2)

    # trimming controls
    parser.add_argument('--start-offset', '-o', type=int, default=0,
                        help='Skip this many poses from the start of the incoming trajectory')
    parser.add_argument('--trim', '-t', type=int, default=0,
                        help='Skip this many poses from the end of the incoming trajectory')
    parser.add_argument('--max-length', '--ml', '-l', type=int, default=0,
                        help='Cap the final number of poses after processing (0 = unlimited)')

    parser.add_argument('--zmq-port', type=int, default=242424)
    parser.add_argument('--urdf-path', '-u', type=str, default=None)
    parser.add_argument('--preview-only', action='store_true')

    parser.add_argument("--no-wait-server", action="store_true", help="Do not wait for the joint action server to start (for debugging without ROS).")

    args = parser.parse_args()

    node = TiagoTrajectoryController(
        arm_side=args.side, unsafe=args.unsafe,
        merge_eps=args.merge, smooth=args.smooth, resample=args.resample,
        v_max=args.v_max, a_max=args.a_max, dt_min=args.dt_min,
        exec_delay=args.exec_delay, urdf_path=args.urdf_path, zmq_port=args.zmq_port,
        start_offset=args.start_offset, trim_end=args.trim, max_length=args.max_length,
        no_wait_server=args.no_wait_server
    )

    rospy.loginfo("ZMQ usage:\n  SEND:  {'cmd':'send','joint_names':[...],'data':[[...],...],"
                  " 'start_offset':K,'trim':T,'max_length':L}\n"
                  "  CANCEL: {'cmd':'cancel'}    HALT: {'cmd':'halt'}")
    rospy.spin()


if __name__ == '__main__':
    main()
