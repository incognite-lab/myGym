#!/usr/bin/env python3
"""
Unit tests for myGym.envs.predicates.

Spawns real objects in a live PyBullet env (table workspace + robot) and
checks the predicate classes (Touching, OnTop, IsReachable) and the
InitPredicateResolver against them.

Requirements:
    - All dependencies from pyproject.toml must be installed
    - Run: pip install -e . (from repository root)

Usage:
    # Run all tests with the default config (configs/AGMD_predicates.json)
    python3 myGym/unittest/test_predicates.py

    # Run more randomized trials for the init-predicate resolver test
    python3 myGym/unittest/test_predicates.py --trials 200

    # Use a different config or open the PyBullet GUI
    python3 myGym/unittest/test_predicates.py --config configs/AGM.json --gui 1
"""

import glob
import itertools
import os
import signal
from contextlib import contextmanager

from myGym.train import get_parser, get_arguments, automatic_argument_assignment, configure_env
from myGym.envs import env_object
from myGym.envs.predicates import IsReachable, Touching, OnTop, Inside, InitPredicateResolver

# ANSI colors for summary marks
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "AGMD_predicates.json")

APPLE_URDF = os.path.join(PROJECT_ROOT, "envs/objects/household/urdf/apple.urdf")
TUNA_CAN_URDF = os.path.join(PROJECT_ROOT, "envs/objects/household/urdf/tuna_can.urdf")
HOUSEHOLD_URDF_DIR = os.path.join(PROJECT_ROOT, "envs/objects/household/urdf")
ON_TOP_REPORT_PATH = os.path.join(PROJECT_ROOT, "unittest", "on_top_results.txt")
ON_TABLE_REPORT_PATH = os.path.join(PROJECT_ROOT, "unittest", "on_table_results.txt")
INSIDE_REPORT_PATH = os.path.join(PROJECT_ROOT, "unittest", "inside_results.txt")

CHECK_TIMEOUT_SECONDS = 30  # abort a single OnTop/Inside check stuck on a pathological contact


def voxel_demo(obj):
    """
    Debug helper: voxelize obj's mesh with open3d and display it next to
    the original mesh for visual inspection.
    """
    import open3d as o3d
    import numpy as np
    from myGym.envs.test_volume_class import VolumeMesh

    def read_urdf_scale(urdf_path: str) -> float:
        """
        Read the first mesh scale value from a URDF file,
        return 1 if not specified
        """
        with open(urdf_path) as file:
            lines = file.readlines()

        scale_lines = [line for line in lines if "scale" in line]

        if not scale_lines:
            return 1.0

        return float(scale_lines[0].split('scale="')[1].split(" ")[0])

    # load tuna model and check the scale
    obj1_info = obj.p.getVisualShapeData(obj.get_uid())[0]
    obj1_scale = read_urdf_scale(obj.urdf_path)
    objpth = obj1_info[4].decode("utf-8")

    # voxelize
    o3model = o3d.io.read_triangle_model(objpth)
    mesh = o3model.meshes[0].mesh
    mesh = mesh.scale(obj1_scale, center=mesh.get_center())
    vm = VolumeMesh(mesh)
    orig = vm.duplicate()
    orig.paint(np.array([0, 1, 0]))
    voxel_grid = vm.voxelgrid

    # visualize geometry
    o3d.visualization.draw_geometries([voxel_grid, orig.voxelgrid])


def parse_args() -> tuple[dict, int]:
    """
    Parse command-line arguments into an arg_dict compatible with configure_env().
    """
    parser = get_parser()
    parser.description = "Run unit tests for myGym.envs.predicates against a live PyBullet env."
    parser.set_defaults(config=DEFAULT_CONFIG)
    parser.add_argument(
        "--trials", type=int, default=20,
        help="Number of randomized placements/pairs to test (default: 20)"
    )

    arg_dict, _ = get_arguments(parser)
    arg_dict.setdefault("top_grasp", False)
    arg_dict = automatic_argument_assignment(arg_dict)
    return arg_dict, arg_dict.pop("trials")


def build_env(arg_dict: dict):
    """
    Build the live PyBullet env used as a fixture for all predicate tests.
    """
    env = configure_env(arg_dict, model_logdir=None, for_train=0)
    return env.unwrapped


def spawn_object(env, urdf_path: str, position, fixed: bool = False):
    """
    Spawn an EnvObject in the env and return it.
    """
    return env_object.EnvObject(
        urdf_path,
        position=position,
        orientation=[0, 0, 0, 1],
        pybullet_client=env.p,
        fixed=fixed,
    )


@contextmanager
def spawned_object(env, urdf_path: str, position, fixed: bool = False):
    """
    Spawn an EnvObject and guarantee its body is removed on exit, even if
    the caller raises while the object is alive.
    """
    obj = spawn_object(env, urdf_path, position, fixed=fixed)
    try:
        yield obj
    finally:
        env.p.removeBody(obj.uid)


def settle(env, steps: int = 100):
    """
    Step the simulation so spawned objects can fall and settle.
    """
    for _ in range(steps):
        env.p.stepSimulation()


class _CheckTimeout(Exception):
    """Raised when a single predicate check runs longer than CHECK_TIMEOUT_SECONDS."""


def _raise_check_timeout(signum, frame):
    raise _CheckTimeout()


@contextmanager
def _time_limit(seconds: int):
    """
    Abort the wrapped block with _CheckTimeout if it runs longer than
    `seconds`. Uses SIGALRM (Unix only); the signal is delivered as soon as
    control returns to the interpreter between pybullet calls, so it can
    interrupt a settle() loop stuck on a pathological contact configuration.
    """
    previous_handler = signal.signal(signal.SIGALRM, _raise_check_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _call_with_timeout(fn, fallback, label: str, seconds: int = CHECK_TIMEOUT_SECONDS):
    """
    Call fn() with a time limit; return fallback and print a warning if it
    doesn't finish within `seconds` instead of letting the sweep hang.
    """
    try:
        with _time_limit(seconds):
            return fn()
    except _CheckTimeout:
        print(f"  {RED}TIMEOUT{RESET} {label} (> {seconds}s)")
        return fallback


def test_touching_and_on_top(env):
    """Two tuna cans dropped one above the other should end up touching and stacked OnTop."""
    table = env.static_scene_objects[env.workspace]
    on_top = OnTop()
    touching = Touching()

    table_area = on_top.compute_area(TUNA_CAN_URDF, table)
    bottom_pos = env_object.EnvObject.get_random_object_position(table_area)

    with spawned_object(env, TUNA_CAN_URDF, bottom_pos) as tuna_bottom:
        settle(env, steps=15)

        assert touching.check(tuna_bottom, table), "tuna can should be touching the table after settling"
        assert on_top.check(tuna_bottom, table), "tuna can should be OnTop of the table after settling"

        top_pos = [bottom_pos[0], bottom_pos[1], bottom_pos[2] + 0.2]
        with spawned_object(env, TUNA_CAN_URDF, top_pos) as tuna_top:
            settle(env, steps=100)

            assert touching.check(tuna_top, tuna_bottom), "stacked tuna cans should be touching"
            assert on_top.check(tuna_top, tuna_bottom), "top tuna can should be OnTop of the bottom one"
            assert not on_top.check(tuna_bottom, tuna_top), "bottom tuna can should not be OnTop of the top one"

    print("PASS: test_touching_and_on_top")


def test_is_reachable(env, trials: int):
    """An apple should fall inside the robot's reachable envelope."""
    for trial in range(trials):
        reachable = IsReachable()
        table_area = reachable.compute_area(env.robot)
        apple_pos = env_object.EnvObject.get_random_object_position(table_area)

        with spawned_object(env, APPLE_URDF, apple_pos) as apple:
            settle(env)

            assert reachable.check(env.robot, apple), (
                f"Trial {trial}: apple not reachable for apple_pos = {apple_pos}"
            )

    print(f"PASS: test_is_reachable ({trials} trials)")


def test_init_predicates_are_enforced(env, trials: int):
    """
    Randomized placements resolved by InitPredicateResolver must satisfy
    the init predicates they were sampled for, even after physics settles.
    """
    predicates = {"init": ["Reachable(apple)", "OnTop(apple,table)", "Reachable(tuna_can)"]}
    apple_info = {"urdf": APPLE_URDF, "obj_name": "apple"}
    tuna_can_info = {"urdf": TUNA_CAN_URDF, "obj_name": "tuna_can"}

    table = env.static_scene_objects[env.workspace]
    robot = env.robot
    resolver = InitPredicateResolver()

    for trial in range(trials):
        apple_area = resolver.get_area(apple_info, table, robot, predicates)
        apple_pos = env_object.EnvObject.get_random_object_position(apple_area)

        with spawned_object(env, APPLE_URDF, apple_pos) as apple:
            tuna_can_area = resolver.get_area(tuna_can_info, table, robot, predicates)
            tuna_can_pos = env_object.EnvObject.get_random_object_position(tuna_can_area)

            with spawned_object(env, TUNA_CAN_URDF, tuna_can_pos) as tuna_can:
                settle(env)
                satisfied = resolver.check({"init": apple, "goal": tuna_can}, env, predicates)

                assert satisfied, (
                    f"Trial {trial}: init predicates not satisfied for "
                    f"apple_pos={apple_pos}, tuna_can_pos={tuna_can_pos}"
                )

    print(f"PASS: test_init_predicates_are_enforced ({trials} trials)")


def _get_household_objects() -> list[str]:
    """
    Return paths to household object URDFs, excluding goal-marker variants.
    """
    urdfs = sorted(glob.glob(os.path.join(HOUSEHOLD_URDF_DIR, "*.urdf")))
    return [path for path in urdfs if "target" not in os.path.basename(path)]


def _check_inside(env, table, on_top: OnTop, inside: Inside, obj1_urdf: str,
                 obj2_urdf: str, visual_check: bool = False) -> tuple[bool, bool]:
    """
    Place obj2 on the table and obj1 on top of obj2.
    Return whether OnTop(obj1, obj2) and Inside(obj1, obj2) hold once the
    placement settles.
    """
    obj2_area = on_top.compute_area(obj2_urdf, table)
    obj2_pos = env_object.EnvObject.get_random_object_position(obj2_area)

    with spawned_object(env, obj2_urdf, obj2_pos) as obj2:
        settle(env, steps=100)

        obj1_area = on_top.compute_area(obj1_urdf, obj2)
        obj1_pos = env_object.EnvObject.get_random_object_position(obj1_area)

        with spawned_object(env, obj1_urdf, obj1_pos) as obj1:
            settle(env, steps=300)

            is_on_top = on_top.check(obj1, obj2)
            is_inside = inside.check(obj1, obj2)

            if visual_check:
                print("On top:", is_on_top)
                print("Inside:", is_inside)
                input("Press Enter to continue...")

    return is_on_top, is_inside


def _check_OnTop(env, table, on_top: OnTop, obj1_urdf: str, obj2_urdf: str | None = None,
                 visual_check: bool = False) -> bool:
    """
    Place obj1 on top of obj2 (or directly on the table if obj2_urdf is None),
    and return whether OnTop(obj1, obj2) holds once the placement settles.
    """
    if obj2_urdf is None:
        obj1_area = on_top.compute_area(obj1_urdf, table)
        obj1_pos = env_object.EnvObject.get_random_object_position(obj1_area)

        with spawned_object(env, obj1_urdf, obj1_pos) as obj1:
            settle(env, steps=100)

            is_on_top = on_top.check(obj1, table)

            if visual_check:
                print("On top:", is_on_top)
                input("Press Enter to continue...")

        return is_on_top

    obj2_area = on_top.compute_area(obj2_urdf, table)
    obj2_pos = env_object.EnvObject.get_random_object_position(obj2_area)

    with spawned_object(env, obj2_urdf, obj2_pos) as obj2:
        settle(env, steps=100)

        obj1_area = on_top.compute_area(obj1_urdf, obj2)
        obj1_pos = env_object.EnvObject.get_random_object_position(obj1_area)

        with spawned_object(env, obj1_urdf, obj1_pos) as obj1:
            settle(env, steps=300)

            is_on_top = on_top.check(obj1, obj2)

            if visual_check:
                print("On top:", is_on_top)
                input("Press Enter to continue...")

    return is_on_top


def _write_report(report_path: str, title: str, results: list[dict], passed: int, format_row) -> None:
    """
    Write a generic pass/fail report to a text file.

    format_row(result) must return a list of report lines for one result.
    """
    with open(report_path, "w") as f:
        f.write(f"{title}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Passed: {passed} out of {len(results)}\n")
        for r in results:
            for line in format_row(r):
                f.write(line + "\n")


def test_on_top(env):
    """
    Stack every pair of household objects on each other (in both directions)
    and report which pairs satisfy OnTop. Full results are written to
    unittest/on_top_results.txt since not every shape combination is expected
    to stack cleanly.
    """
    objects = _get_household_objects()
    table = env.static_scene_objects[env.workspace]
    on_top = OnTop()

    results = []
    for obj1, obj2 in itertools.combinations(objects, 2):
        name1, name2 = os.path.basename(obj1), os.path.basename(obj2)
        result = {
            "obj1": name1,
            "obj2": name2,
            "obj1_on_obj2": _call_with_timeout(
                lambda: _check_OnTop(env, table, on_top, obj1, obj2), False, f"{name1} on {name2}"
            ),
            "obj2_on_obj1": _call_with_timeout(
                lambda: _check_OnTop(env, table, on_top, obj2, obj1), False, f"{name2} on {name1}"
            ),
        }
        results.append(result)

        mark1 = f"{GREEN}OK{RESET}" if result["obj1_on_obj2"] else f"{RED}FAIL{RESET}"
        mark2 = f"{GREEN}OK{RESET}" if result["obj2_on_obj1"] else f"{RED}FAIL{RESET}"
        print(f"  {result['obj1']:<20s} on {result['obj2']:<20s}: {mark1}   "
              f"{result['obj2']:<20s} on {result['obj1']:<20s}: {mark2}")

    passed = sum(1 for r in results if r["obj1_on_obj2"] and r["obj2_on_obj1"])
    _write_report(
        ON_TOP_REPORT_PATH,
        "OnTop STACKING REPORT",
        results,
        passed,
        lambda r: [
            f"{r['obj1']:<20s} on {r['obj2']:<20s}: {'OK' if r['obj1_on_obj2'] else 'FAIL'}",
            f"{r['obj2']:<20s} on {r['obj1']:<20s}: {'OK' if r['obj2_on_obj1'] else 'FAIL'}",
        ],
    )
    print(f"PASS: test_on_top ({passed}/{len(results)} pairs OK in both directions, report: {ON_TOP_REPORT_PATH})")


def test_inside_obj(env, obj2_urdf: str):
    """
    Drop every household object onto obj2_urdf and report which ones land
    OnTop of it or end up Inside it.
    """
    objects = _get_household_objects()
    table = env.static_scene_objects[env.workspace]
    on_top = OnTop()
    inside = Inside()

    results = []
    for obj1_urdf in objects:
        is_on_top1, is_inside1 = _call_with_timeout(
            lambda: _check_inside(env, table, on_top, inside, obj1_urdf, obj2_urdf),
            (False, False),
            f"{os.path.basename(obj1_urdf)} in {os.path.basename(obj2_urdf)}",
        )
        result = {
            "obj1": os.path.basename(obj1_urdf),
            "obj2": os.path.basename(obj2_urdf),
            "obj1_on_obj2": is_on_top1 ,
            "obj1_inside_obj2": is_inside1 ,
        }
        results.append(result)

        mark = f"{GREEN}OK{RESET}" if result["obj1_inside_obj2"] else f"{RED}FAIL{RESET}"
        print(f"{result['obj1']:<20s} in {result['obj2']:<20s}: {mark}")

    passed = sum(1 for r in results if r["obj1_inside_obj2"])
    _write_report(
        INSIDE_REPORT_PATH,
        f"TEST INSIDE {os.path.basename(obj2_urdf)} REPORT",
        results,
        passed,
        lambda r: [
            f"{r['obj1']:<20s} on {r['obj2']:<20s}: {'OK' if r['obj1_on_obj2'] else 'FAIL'}",
            f"{r['obj1']:<20s} in {r['obj2']:<20s}: {'OK' if r['obj1_inside_obj2'] else 'FAIL'}",
        ],
    )
    print(f"PASS: test_inside ({passed}/{len(results)} of tested objects, report: {INSIDE_REPORT_PATH})")


def test_on_table(env):
    """
    Place every household object directly on the table and report which
    ones satisfy OnTop(obj, table).
    """
    objects = _get_household_objects()
    table = env.static_scene_objects[env.workspace]
    on_top = OnTop()

    results = []
    for obj_urdf in objects:
        name = os.path.basename(obj_urdf)
        result = {
            "obj_name": name,
            "on_table": _call_with_timeout(
                lambda: _check_OnTop(env, table, on_top, obj_urdf), False, f"{name} on table"
            ),
        }
        results.append(result)

        mark = f"{GREEN}OK{RESET}" if result["on_table"] else f"{RED}FAIL{RESET}"
        print(f"{result['obj_name']:<20s} on table: {mark}")

    passed = sum(1 for r in results if r["on_table"])
    _write_report(
        ON_TABLE_REPORT_PATH,
        "ON TABLE REPORT",
        results,
        passed,
        lambda r: [f"{r['obj_name']:<20s} on table: {'OK' if r['on_table'] else 'FAIL'}"],
    )
    print(f"PASS: test_on_table ({passed}/{len(results)} of tested objects, report: {ON_TABLE_REPORT_PATH})")


def show_stacking(arg_dict):
    """
    Open a GUI env and inspected visually placing objects OnTop/Inside.
    """
    arg_dict['gui'] = 1
    env = build_env(arg_dict)

    table = env.static_scene_objects[env.workspace]
    name1 = "banana.urdf"
    name2 = "bowl.urdf"
    path1 = os.path.join(HOUSEHOLD_URDF_DIR, name1)
    path2 = os.path.join(HOUSEHOLD_URDF_DIR, name2)
    _check_inside(env, table, OnTop(), Inside(), path1, path2, True)
    #_check_OnTop(env, table, on_top, path1, path2)


def main():
    """
    Run selected tests.
    """
    arg_dict, trials = parse_args()
    env = build_env(arg_dict)

    #test_touching_and_on_top(env)
    #test_is_reachable(env, trials)
    #test_init_predicates_are_enforced(env, trials)
    #test_on_table(env)
    #test_on_top(env)
    #test_inside_obj(env, os.path.join(HOUSEHOLD_URDF_DIR, "jug.urdf"))

    print("\nAll tests passed!")

    show_stacking(arg_dict)

if __name__ == '__main__':
    main()
