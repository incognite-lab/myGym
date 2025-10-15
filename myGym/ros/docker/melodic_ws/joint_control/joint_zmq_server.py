#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
TIAGo trajectory bridge (Python 2.7, ROS Melodic)

- Receives joint trajectories over ZeroMQ (cloudpickle or JSON)
- Merges close points, smooths optionally, and re-times by joint-space distance
- Sends to FollowJointTrajectory action (safe/unsafe arm controllers)
- Robust preemption via SimpleActionClient, plus "halt" that freezes in place
- RViz preview (DisplayTrajectory + Markers)
- Approximate self-collision check using URDF + KDL + bounding spheres
"""

from __future__ import print_function, division
import sys, os, argparse, json, threading, math, random, time
import numpy as np

import rospy
import actionlib
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

# SciPy is optional; we fall back gracefully if missing
try:
    from scipy.interpolate import splprep, splev
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ZeroMQ
import zmq
import cloudpickle as cpl
import tty, termios


# ---------- math & utils ----------
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


def _angle_diff(a, b):
    """Shortest signed distance a->b for rotational joints (assumes radians)."""
    d = (b - a + math.pi) % (2.0 * math.pi) - math.pi
    return d


def joint_distance(p, q):
    """Euclidean distance in joint space (assume radians)."""
    p = np.asarray(p); q = np.asarray(q)
    return float(np.linalg.norm(q - p, ord=2))


def merge_consecutive(points, eps):
    """
    Greedy *sequential* merge of consecutive points that are closer than eps.
    Preserves order and shape.
    """
    if not points:
        return []
    out = [np.asarray(points[0], dtype=float)]
    run_sum = out[0].copy()
    run_n = 1
    for i in range(1, len(points)):
        cur = np.asarray(points[i], dtype=float)
        if joint_distance(out[-1], cur) < eps:
            # merge into running average (keeps plateau but avoids duplicates)
            run_sum += cur
            run_n += 1
            out[-1] = run_sum / float(run_n)
        else:
            # reset run
            run_sum = cur.copy()
            run_n = 1
            out.append(cur)
    return [p.tolist() for p in out]


def catmull_rom(points, upsample_steps=1):
    """
    Lightweight C^1 smoothing.
    Returns same number of points if upsample_steps==1; otherwise densifies.
    """
    P = [np.asarray(p, dtype=float) for p in points]
    if len(P) <= 2 or upsample_steps <= 1:
        return [p.tolist() for p in P]
    Q = []
    for i in range(len(P)):
        p0 = P[i - 1] if i - 1 >= 0 else P[i]
        p1 = P[i]
        p2 = P[i + 1] if i + 1 < len(P) else P[i]
        p3 = P[i + 2] if i + 2 < len(P) else p2
        # Emit p1 only when i > 0 to avoid duplicates
        for s in range(upsample_steps):
            t = float(s) / float(upsample_steps)
            t2, t3 = t * t, t * t * t
            # Catmull-Rom spline basis
            a = 2 * p1
            b = -p0 + p2
            c = 2 * p0 - 5 * p1 + 4 * p2 - p3
            d = -p0 + 3 * p1 - 3 * p2 + p3
            q = 0.5 * (a + b * t + c * t2 + d * t3)
            if (i == 0 and s == 0) or s > 0:
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
    # parameter via arc-length
    arc = [0.0]
    for i in range(1, N):
        arc.append(arc[-1] + joint_distance(P[i - 1], P[i]))
    arc = np.asarray(arc, dtype=float)
    if math.fabs(arc[-1]) < 1e-6:
        return [p.tolist() for p in P]
    u = arc / arc[-1]

    if resample is None:
        resample = N
    u_new = np.linspace(0., 1., int(resample))

    if SCIPY_OK:
        coord = [P[:, j] for j in range(D)]
        # map smoothing 0..1 to s parameter ~ [0, N]
        s = float(smoothing) * float(N)
        tck, _ = splprep(coord, u=u, s=s, k=3)
        sm = splev(u_new, tck)
        Q = np.vstack(sm).T
        return [q.tolist() for q in Q]
    else:
        # lightweight fallback
        up = int(max(1, round(float(resample) / float(N-1))))
        Q = catmull_rom(P, upsample_steps=up)
        if len(Q) != resample:
            # simple resample
            idx = np.linspace(0, len(Q)-1, resample)
            QQ = []
            for t in idx:
                i = int(math.floor(t))
                a = Q[i]
                b = Q[min(i + 1, len(Q) - 1)]
                alpha = t - i
                QQ.append((1.0 - alpha)*np.asarray(a) + alpha * np.asarray(b))
            Q = [q.tolist() for q in QQ]
        return Q


def retime_by_distance(points, v_max=0.7, a_max=1.5, dt_min=0.08, t0=0.0):
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
                 exec_delay=0.2, urdf_path=None, zmq_port=242424):
        """
        merge_eps: radians threshold for sequential merging
        smooth: 0..1 smoothing weight
        resample: 0 (keep length) or int points
        v_max, a_max, dt_min: timing params
        exec_delay: seconds to delay header.stamp
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

        rospy.init_node('tiago_traj_bridge', anonymous=True)

        # Choose controller names (safe/unsafe)
        base_ns = ('/safe_' if self.safe else '/') + 'arm_%s_controller' % self.arm_side
        self.action_name = base_ns + '/follow_joint_trajectory'
        self.topic_cmd = base_ns + '/command'  # topic interface

        # Action client (SimpleActionClient for robust preempt behavior)
        self.client = actionlib.SimpleActionClient(self.action_name, FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for action server: %s", self.action_name)
        if not self.client.wait_for_server(rospy.Duration(15.0)):
            rospy.logerr("Action server not available: %s", self.action_name)
            raise RuntimeError("No action server")

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
                ok, tree = treeFromParam('/robot_description')
                if not ok:
                    raise RuntimeError("Failed to parse URDF from /robot_description")
                robot = None  # not needed
            else:
                robot = URDF.from_parameter_server()
                ok, tree = treeFromUrdfModel(robot)
                if not ok:
                    raise RuntimeError("Failed to build KDL tree from URDF")
        else:
            robot = URDF.from_xml_file(urdf_path)
            ok, tree = treeFromUrdfModel(robot)
            if not ok:
                raise RuntimeError("Failed to build KDL tree from URDF file")

        self.base_link = 'torso_lift_link'
        self.ee_link = 'arm_%s_7_link' % self.arm_side
        self.chain = tree.getChain(self.base_link, self.ee_link)
        self.fk = kdl.ChainFkSolverPos_recursive(self.chain)

        # prebuild link names for arm chain and a conservative sphere per link
        self.arm_links = []
        for i in range(self.chain.getNrOfSegments()):
            self.arm_links.append(self.chain.getSegment(i).getName())
        self.bounds = self._build_bounding_spheres(robot)

        # ZMQ server
        self.server = ZmqServer(self._on_request, port=zmq_port)
        rospy.on_shutdown(self._on_shutdown)

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
         - {"cmd":"send","joint_names":[...],"data":[[...],...], "merge":eps, "smooth":w, "resample":N, "v_max":..., "a_max":..., "dt_min":...}
         - {"cmd":"cancel"}
         - {"cmd":"halt"}  # cancel + hold at current pos
        Backward compatible with old dicts having 'joint_names' and 'data'.
        """
        try:
            cmd = data.get('cmd', None)
        except Exception:
            cmd = None

        if cmd in (None, 'send'):
            joint_names = data.get('joint_names', None)
            positions = data.get('data', None)
            if joint_names is None or positions is None:
                # maybe old payload sent with cloudpickle
                joint_names = data['joint_names']
                positions = data['data']
            # sanity
            if not positions or not joint_names:
                return {"ok": False, "err": "empty trajectory"}
            # normalize names (your original cleanup kept here)
            names = []
            for nm in joint_names:
                s = str(nm).replace('_rjoint', '_joint').replace("b'", '').replace("'", '')
                names.append(s)
            # optional per-request overrides
            merge = float(data.get('merge', self.merge_eps))
            smooth = float(data.get('smooth', self.smooth_w))
            resample = int(data.get('resample', self.resample))
            v_max = float(data.get('v_max', self.v_max))
            a_max = float(data.get('a_max', self.a_max))
            dt_min = float(data.get('dt_min', self.dt_min))
            preview_only = bool(data.get('preview_only', False))

            ok, warn = self.send_trajectory(names, positions,
                                            merge=merge, smooth=smooth,
                                            resample=resample,
                                            v_max=v_max, a_max=a_max, dt_min=dt_min,
                                            preview_only=preview_only)
            return {"ok": ok, "warning": warn}

        elif cmd == 'cancel':
            self.cancel_goal()
            return {"ok": True}
        elif cmd == 'halt':
            self.halt()
            return {"ok": True}
        else:
            return {"ok": False, "err": "unknown cmd: %r" % cmd}

    # ---- main features ----
    def build_traj(self, joint_names, points, times):
        jt = JointTrajectory()
        jt.header.stamp = rospy.Time.now() + rospy.Duration(self.exec_delay)
        jt.header.frame_id = self.base_link
        jt.joint_names = list(joint_names)
        for p, t in zip(points, times):
            pt = JointTrajectoryPoint()
            pt.positions = list(map(float, p))
            pt.time_from_start = rospy.Duration(float(t))
            jt.points.append(pt)
        return jt

    def visualize(self, traj, collisions):
        # DisplayTrajectory (no MoveIt requirement)
        disp = DisplayTrajectory()
        disp.trajectory.append(RobotTrajectory(joint_trajectory=traj))
        with self.joint_state_lock:
            if self.joint_state is not None:
                disp.trajectory_start.joint_state = self.joint_state
        self.disp_pub.publish(disp)

        # EE path & collision markers
        marr = MarkerArray()
        line = Marker()
        line.header.frame_id = self.base_link
        line.header.stamp = rospy.Time.now()
        line.ns = 'ee_path'; line.id = 0
        line.type = Marker.LINE_STRIP; line.action = Marker.ADD
        line.scale.x = 0.006
        line.color = ColorRGBA(0.1, 0.9, 0.1, 1.0)
        marr.markers.append(line)

        for k, pt in enumerate(traj.points):
            # fk to ee (quick demo path)
            jarr = kdl.JntArray(len(pt.positions))
            for i, q in enumerate(pt.positions): jarr[i] = q
            frame = kdl.Frame()
            self.fk.JntToCart(jarr, frame)
            p = frame.p
            gp = Point(p.x(), p.y(), p.z())
            marr.markers[0].points.append(gp)

            # draw sample spheres for flagged collisions
            if collisions and collisions[k]:
                m = Marker()
                m.header = line.header
                m.ns = 'coll_warn'; m.id = k+1
                m.type = Marker.SPHERE; m.action = Marker.ADD
                m.pose.position = gp; m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 0.03
                m.color = ColorRGBA(1.0, 0.2, 0.2, 0.9)
                marr.markers.append(m)
        self.marker_pub.publish(marr)

    def check_self_collision(self, joint_names, points):
        """
        Approximate, arm-only, sphere-vs-sphere self-collision.
        Returns a boolean list per point (True if any collision suspected).
        """
        # Build FK solvers per segment index that correspond to each joint order
        # Our KDL chain is arm-only starting at torso_lift_link; positions are in controller joint order
        # We'll compute link frames by forward kinematics at each point
        if len(points) == 0:
            return []

        # Precompute which pairs to test: non-adjacent arm links that have spheres
        pairs = []
        indices = []
        for i, ln_i in enumerate(self.arm_links):
            if ln_i not in self.bounds: continue
            for j, ln_j in enumerate(self.arm_links):
                if j <= i + 1: continue  # skip same & adjacent
                if ln_j not in self.bounds: continue
                pairs.append((ln_i, ln_j))
                indices.append((i, j))

        result = []
        jnt_size = len(points[0])
        for k, q in enumerate(points):
            jarr = kdl.JntArray(jnt_size)
            for i, val in enumerate(q): jarr[i] = float(val)
            # Forward propagate along chain to get each segment frame
            frames = []
            f = kdl.Frame()
            for seg_idx in range(self.chain.getNrOfSegments()):
                self.fk.JntToCart(jarr, f, seg_idx+1)
                frames.append(kdl.Frame(f))
            any_collision = False
            for (ln_i, ln_j), (i, j) in zip(pairs, indices):
                c_i, r_i = self.bounds.get(ln_i, ((0,0,0), 0.0))
                c_j, r_j = self.bounds.get(ln_j, ((0,0,0), 0.0))
                # Sphere centers in link frames -> base_link frame
                p_i = frames[i].p + frames[i].M * kdl.Vector(c_i[0], c_i[1], c_i[2])
                p_j = frames[j].p + frames[j].M * kdl.Vector(c_j[0], c_j[1], c_j[2])
                dx = p_i.x()-p_j.x(); dy = p_i.y()-p_j.y(); dz = p_i.z()-p_j.z()
                d2 = dx*dx + dy*dy + dz*dz
                tol = 1e-6
                if d2 < (r_i + r_j - 0.01)*(r_i + r_j - 0.01) - tol:  # small margin
                    any_collision = True
                    break
            result.append(any_collision)
        return result

    def _build_bounding_spheres(self, robot):
        """
        Very conservative bounding spheres per arm link from URDF collision geometry.
        If URDF not provided to this method, fall back to default radii.
        """
        bounds = {}
        default_r = 0.06  # meters, conservative
        for ln in self.arm_links:
            bounds[ln] = ((0.0, 0.0, 0.0), default_r)
        try:
            if robot is None:
                return bounds
            for ln in self.arm_links:
                if ln not in robot.link_map:
                    continue
                L = robot.link_map[ln]
                if L.collision and L.collision.geometry:
                    g = L.collision.geometry
                    # quick approximate radius from primitives; meshes use default
                    if hasattr(g, 'radius') and g.radius:
                        r = float(g.radius)
                    elif hasattr(g, 'size') and g.size:
                        # box -> half-diagonal
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

    def send_trajectory(self, joint_names, positions,
                        merge=None, smooth=None, resample=None,
                        v_max=None, a_max=None, dt_min=None,
                        preview_only=False):
        # defaults
        if merge is None: merge = self.merge_eps
        if smooth is None: smooth = self.smooth_w
        if resample is None: resample = self.resample
        if v_max is None: v_max = self.v_max
        if a_max is None: a_max = self.a_max
        if dt_min is None: dt_min = self.dt_min

        # preprocess
        pts = [list(map(float, p)) for p in positions]
        if merge > 0.0:
            pts = merge_consecutive(pts, merge)
        if smooth > 0.0 or (resample and resample > 0):
            pts = smooth_points(pts, smoothing=smooth, resample=(resample or len(pts)))

        # times
        times = retime_by_distance(pts, v_max=v_max, a_max=a_max, dt_min=dt_min, t0=dt_min)

        # build trajectory
        traj = self.build_traj(joint_names, pts, times)

        # approximate collision check
        collisions = self.check_self_collision(joint_names, pts)
        n_bad = sum(1 for c in collisions if c)
        warn = None
        if n_bad > 0:
            warn = "approx_self_collision_suspected_in_%d_points" % n_bad
            rospy.logwarn("Approx. self-collision suspected in %d points (arm only).", n_bad)

        # preview
        self.visualize(traj, collisions)
        if preview_only:
            return True, warn

        key = read_char("Execute goal? (y/n): ")
        while key not in ['y', 'n']:
            key = read_char("Wrong input {}. Execute goal? (y/n): ".format(key))
        if key == 'n':
            rospy.logwarn("Aborting goal.")
            return

        # send goal
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = traj
        self.client.send_goal(goal)
        rospy.loginfo("Goal sent: %d waypoints (v_max=%.2f, a_max=%.2f, dt_min=%.2f)",
                      len(pts), v_max, a_max, dt_min)
        return True, warn

    def cancel_goal(self):
        try:
            self.client.cancel_goal()
            rospy.loginfo("cancel_goal() sent.")
        except Exception as e:
            rospy.logwarn("cancel_goal failed: %s", e)

    def halt(self, hold_dt=0.15):
        """
        Cancel and then send a one-point 'hold' goal at current position
        for a deterministic freeze-in-place.
        """
        self.cancel_goal()
        # read current joint positions in the controller's joint order
        names = None
        # We can get them from the last goal if needed; easier: infer from controller
        # Typical TIAGo arm joint names:
        names = ['arm_%s_%d_joint' % (self.arm_side, i) for i in range(1, 8)]
        cur = self._current_positions(names)
        if cur is None:
            rospy.logwarn("Cannot read /joint_states to build hold goal; aborting halt.")
            return
        jt = self.build_traj(names, [cur], [hold_dt])
        goal = FollowJointTrajectoryGoal(trajectory=jt)
        self.client.send_goal(goal)
        rospy.loginfo("Hold goal sent at current joint positions.")

    def _on_shutdown(self):
        try:
            self.cancel_goal()
        except Exception:
            pass
        try:
            self.server.close()
        except Exception:
            pass


# ---- entry point ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--side', choices=['left', 'right'], default='right')
    parser.add_argument('--unsafe', action='store_true')
    parser.add_argument('--merge', type=float, default=0.0)
    parser.add_argument('--smooth', type=float, default=0.0)
    parser.add_argument('--resample', type=int, default=0)
    parser.add_argument('--v-max', type=float, default=0.7)
    parser.add_argument('--a-max', type=float, default=1.5)
    parser.add_argument('--dt-min', type=float, default=0.08)
    parser.add_argument('--exec-delay', type=float, default=0.2)
    parser.add_argument('--zmq-port', type=int, default=242424)
    parser.add_argument('--urdf-path', type=str, default=None)
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()

    node = TiagoTrajectoryController(
        arm_side=args.side, unsafe=args.unsafe,
        merge_eps=args.merge, smooth=args.smooth, resample=args.resample,
        v_max=args.v_max, a_max=args.a_max, dt_min=args.dt_min,
        exec_delay=args.exec_delay, urdf_path=args.urdf_path, zmq_port=args.zmq_port
    )

    # Optional: set stop_trajectory_duration for smoother cancel (controller param)
    # rospy.set_param('/arm_%s_controller/stop_trajectory_duration' % args.side, 0.15)

    rospy.loginfo("ZMQ usage:\n  SEND:  {'cmd':'send','joint_names':[...],'data':[[...],...]}  "
                  "or old cloudpickle dict\n  CANCEL: {'cmd':'cancel'}    HALT: {'cmd':'halt'}")
    rospy.spin()


if __name__ == '__main__':
    main()
