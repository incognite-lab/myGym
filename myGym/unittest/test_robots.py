#!/usr/bin/env python3
import os
import sys
import time
import argparse
import pybullet as p
import pybullet_data
from myGym.utils.helpers import get_robot_dict
import re
import tempfile
import xml.etree.ElementTree as ET

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOTS_ROOT = os.path.join(PROJECT_ROOT, 'envs', 'robots')

TIME_STEP = 1.0 / 240.0
STEPS_PER_TARGET = 480  # ~2 seconds per target
TOLERANCE = 1e-2  # 0.01 rad/m
MAX_FORCE = 100.0
MAX_VELOCITY = 5.0

# ANSI colors for summary marks
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


class _FDOutputCapture:
    """Capture OS-level stdout/stderr (fd 1/2) to suppress PyBullet C++ prints."""
    def __enter__(self):
        import os
        self._os = os
        self._saved_out = os.dup(1)
        self._saved_err = os.dup(2)
        self._tmp_out = tempfile.TemporaryFile(mode='w+b')
        self._tmp_err = tempfile.TemporaryFile(mode='w+b')
        os.dup2(self._tmp_out.fileno(), 1)
        os.dup2(self._tmp_err.fileno(), 2)
        self.out = ""
        self.err = ""
        return self

    def __exit__(self, exc_type, exc, tb):
        os = self._os
        try:
            # Restore fds first
            os.dup2(self._saved_out, 1)
            os.dup2(self._saved_err, 2)
        finally:
            os.close(self._saved_out)
            os.close(self._saved_err)
        # Read captured
        try:
            self._tmp_out.seek(0)
            self.out = self._tmp_out.read().decode('utf-8', errors='ignore')
        finally:
            self._tmp_out.close()
        try:
            self._tmp_err.seek(0)
            self.err = self._tmp_err.read().decode('utf-8', errors='ignore')
        finally:
            self._tmp_err.close()
        # don't suppress exceptions
        return False


def resolve_urdf_path(rel_or_abs: str) -> str:
    # Paths in r_dict start with '/envs/robots/...'; treat as project-relative
    rel = rel_or_abs.lstrip('/')
    abs_path = os.path.join(PROJECT_ROOT, rel)
    return abs_path


def find_links_with_missing_inertia(urdf_path: str):
    """Parse URDF and return link names with missing inertial data (no <inertial>, or missing <mass>/<inertia>)."""
    missing = []
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        for link in root.findall('link'):
            name = link.get('name', '<unnamed>')
            inertial = link.find('inertial')
            if inertial is None:
                missing.append(name)
                continue
            mass_el = inertial.find('mass')
            inertia_el = inertial.find('inertia')
            if mass_el is None or inertia_el is None:
                missing.append(name)
                continue
            # ensure required inertia attributes exist
            req = ['ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz']
            if any(inertia_el.get(k) is None for k in req):
                missing.append(name)
                continue
            # ensure mass value is present and numeric
            try:
                float(mass_el.get('value'))
            except Exception:
                missing.append(name)
    except Exception:
        # If URDF can't be parsed, don't block testing; just return empty to avoid noise
        return []
    return missing


def load_robot(urdf_path: str):
    p.resetSimulation()
    p.setTimeStep(TIME_STEP)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    try:
        with _FDOutputCapture() as cap:
            rid = p.loadURDF(urdf_path, useFixedBase=True)
        return rid
    except Exception:
        try:
            with _FDOutputCapture() as cap:
                rid = p.loadURDF(urdf_path, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
            return rid
        except Exception as ex:
            print(f"[ERROR] Failed to load URDF: {urdf_path}: {ex}")
            return None


def _report_inertial_warnings(raw_log: str):
    # Kept for backward compatibility; inertial issues are now reported via URDF parsing.
    return


def reachable(robot_id: int, joint_index: int, target: float, visualize: bool = False):
    p.setJointMotorControl2(robot_id, joint_index, p.POSITION_CONTROL, targetPosition=target, force=MAX_FORCE)
    last = None
    for _ in range(STEPS_PER_TARGET):
        p.stepSimulation()
        last = p.getJointState(robot_id, joint_index)[0]
        if visualize:
            time.sleep(TIME_STEP)
    err = abs(last - target) if last is not None else float('inf')
    return (err <= TOLERANCE), (last if last is not None else float('nan'))


def check_limits(urdf_path: str, visualize: bool = False):
    rid = load_robot(urdf_path)
    if rid is None:
        return False, ["[LOAD] could not be loaded"]
    failures = []
    nj = p.getNumJoints(rid)
    for j in range(nj):
        ji = p.getJointInfo(rid, j)
        name = ji[1].decode('utf-8') if isinstance(ji[1], (bytes, bytearray)) else str(ji[1])
        jtype = ji[2]
        if jtype == p.JOINT_FIXED:
            continue
        lower = float(ji[8])
        upper = float(ji[9])
        # Skip unlimited joints
        if lower >= upper:
            continue
        # Run lower then upper target tests, and print a single progress line per joint
        ok_l, reached_l = reachable(rid, j, lower, visualize=visualize)
        ok_u, reached_u = reachable(rid, j, upper, visualize=visualize)
        line = f"    {name}: lower {lower:.4f} "
        if ok_l:
            line += f"{GREEN}✔{RESET}"
        else:
            line += f"{RED}✖ (reached {reached_l:.4f}){RESET}"
            failures.append(f"{name} -> lower {lower:.4f}, reached {reached_l:.4f}")
        line += f"  upper {upper:.4f} "
        if ok_u:
            line += f"{GREEN}✔{RESET}"
        else:
            line += f"{RED}✖ (reached {reached_u:.4f}){RESET}"
            failures.append(f"{name} -> upper {upper:.4f}, reached {reached_u:.4f}")
        print(line)
    return (len(failures) == 0), failures


def main():
    parser = argparse.ArgumentParser(description='Test robot URDF joint limit reachability')
    parser.add_argument('--gui', action='store_true', help='Visualize the tests in PyBullet GUI')
    args = parser.parse_args()

    if p.getConnectionInfo().get('isConnected', 0):
        p.disconnect()
    p.connect(p.GUI if args.gui else p.DIRECT)

    # Build list from r_dict only
    rdict = get_robot_dict()
    items = []  # (key, abs_path)
    seen_paths = set()
    for key, info in sorted(rdict.items()):
        raw_path = info.get('path')
        if not raw_path:
            continue
        abs_path = resolve_urdf_path(raw_path)
        if abs_path in seen_paths:
            # Avoid duplicate testing when multiple keys point to same URDF
            continue
        seen_paths.add(abs_path)
        items.append((key, abs_path))

    if not items:
        print("No URDF entries found in r_dict.")
        return

    total = len(items)
    ok_count = 0
    ok_robots = []
    fail_details = {}
    print(f"Testing {total} robot URDFs from r_dict...\n")
    for key, path in items:
        rel = os.path.relpath(path, PROJECT_ROOT)
        if not os.path.exists(path):
            print(f"FAIL {key}: missing file {rel}")
            fail_details[key] = [f"[FILE] missing: {rel}"]
            continue
        # progress: announce robot before testing
        print(f"\nRobot: {key} ({rel})")
        # Inertia testing section
        print("  Inertia testing:")
        missing = find_links_with_missing_inertia(path)
        for lname in missing:
            print(f"    {RED}✖{RESET} {lname} Inertia missing")
        # Joint limit testing section
        print("  Joint limit testing:")
        ok, issues = check_limits(path, visualize=args.gui)
        if ok:
            print(f"OK   {key}: {rel}")
            ok_count += 1
            ok_robots.append(key)
        else:
            print(f"FAIL {key}: {rel}")
            for line in issues:
                print(f"  {line}")
            fail_details[key] = issues
        if not args.gui:
            time.sleep(0.005)

    print("\nSummary:")
    print(f"  OK:   {ok_count}/{total}")
    print(f"  FAIL: {total - ok_count}/{total}")

    # Per-robot results with green OK mark and red mark for problems
    print("\nResults (per robot):")
    for key in sorted(ok_robots):
        # One line per robot with no problems
        print(f"  {GREEN}✔ OK{RESET} {key}")
    for key in sorted(fail_details.keys()):
        # Robot with problems and list of problematic joints
        print(f"  {RED}✖ {key}{RESET}")
        for issue in fail_details[key]:
            # issue already includes joint name and target (lower/upper) vs reached values
            print(f"    - {issue}")

    p.disconnect()


if __name__ == '__main__':
    main()
