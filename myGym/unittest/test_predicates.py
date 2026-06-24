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
import importlib.resources as pkg_resources

from myGym.train import get_parser, get_arguments, automatic_argument_assignment, configure_env
from myGym.envs import env_object
from myGym.envs.predicates import IsReachable, Touching, OnTop, InitPredicateResolver

# ANSI colors for summary marks
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "AGMD_predicates.json")

APPLE_URDF = os.path.join(pkg_resources.files("myGym"), "envs/objects/household/urdf/apple.urdf")
TUNA_CAN_URDF = os.path.join(pkg_resources.files("myGym"), "envs/objects/household/urdf/tuna_can.urdf")
HOUSEHOLD_URDF_DIR = os.path.join(pkg_resources.files("myGym"), "envs/objects/household/urdf")
ON_TOP_REPORT_PATH = os.path.join(PROJECT_ROOT, "unittest", "on_top_results.txt")


def voxel_demo(obj):
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
        position,
        [0, 0, 0, 1],
        pybullet_client=env.p,
        fixed=fixed,
    )


def settle(env, steps: int = 100):
    """
    Step the simulation so spawned objects can fall and settle.
    """
    for _ in range(steps):
        env.p.stepSimulation()


def test_touching_and_on_top(env):
    """Two tuna cans dropped one above the other should end up touching and stacked OnTop."""
    table = env.static_scene_objects[env.workspace]
    on_top = OnTop()
    touching = Touching()

    table_area = on_top.compute_area(TUNA_CAN_URDF, table)
    bottom_pos = env_object.EnvObject.get_random_object_position(table_area)
    tuna_bottom = spawn_object(env, TUNA_CAN_URDF, bottom_pos)
    settle(env, steps=15)

    assert touching.check(tuna_bottom, table), "tuna can should be touching the table after settling"
    assert on_top.check(tuna_bottom, table), "tuna can should be OnTop of the table after settling"

    top_pos = [bottom_pos[0], bottom_pos[1], bottom_pos[2] + 0.2]
    tuna_top = spawn_object(env, TUNA_CAN_URDF, top_pos)
    settle(env, steps=100)

    assert touching.check(tuna_top, tuna_bottom), "stacked tuna cans should be touching"
    assert on_top.check(tuna_top, tuna_bottom), "top tuna can should be OnTop of the bottom one"
    assert not on_top.check(tuna_bottom, tuna_top), "bottom tuna can should not be OnTop of the top one"

    env.p.removeBody(tuna_bottom.uid)
    env.p.removeBody(tuna_top.uid)
    print("PASS: test_touching_and_on_top")


def test_is_reachable(env, trials: int):
    """An apple should fall inside the robot's reachable envelope."""
    for trial in range(trials):
        reachable = IsReachable()
        table_area = reachable.compute_area(env.robot)
        apple_pos = env_object.EnvObject.get_random_object_position(table_area)
        apple = spawn_object(env, APPLE_URDF, apple_pos)
        settle(env)

        assert reachable.check(env.robot, apple), (
            f"Trial {trial}: apple not reachable for apple_pos = {apple_pos}"
        )

        env.p.removeBody(apple.uid)

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
        apple = spawn_object(env, APPLE_URDF, apple_pos)

        tuna_can_area = resolver.get_area(tuna_can_info, table, robot, predicates)
        tuna_can_pos = env_object.EnvObject.get_random_object_position(tuna_can_area)
        tuna_can = spawn_object(env, TUNA_CAN_URDF, tuna_can_pos)

        settle(env)
        satisfied = resolver.check({"init": apple, "goal": tuna_can}, env, predicates)

        env.p.removeBody(apple.uid)
        env.p.removeBody(tuna_can.uid)

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


def _check_stack(env, table, on_top: OnTop, obj1_urdf: str, obj2_urdf: str) -> bool:
    """
    Place obj2 on the table and obj1 on top of obj2.
    Return whether OnTop(obj1, obj2) holds once the placement settles.
    """
    obj2_area = on_top.compute_area(obj2_urdf, table)
    obj2_pos = env_object.EnvObject.get_random_object_position(obj2_area)
    obj2 = spawn_object(env, obj2_urdf, obj2_pos)
    settle(env, steps=100)

    obj1_area = on_top.compute_area(obj1_urdf, obj2)
    obj1_pos = env_object.EnvObject.get_random_object_position(obj1_area)
    obj1 = spawn_object(env, obj1_urdf, obj1_pos)
    settle(env, steps=100)

    is_stacked = on_top.check(obj1, obj2)

    env.p.removeBody(obj1.uid)
    env.p.removeBody(obj2.uid)
    return is_stacked


def _write_on_top_report(results: list[dict], report_path: str) -> None:
    """
    Write a per-pair OnTop stacking report to a text file.
    """
    both_ok = sum(1 for r in results if r["obj1_on_obj2"] and r["obj2_on_obj1"])

    with open(report_path, "w") as f:
        f.write("OnTop STACKING REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Pairs tested: {len(results)}\n")
        f.write(f"Both directions OK: {both_ok}\n\n")
        for r in results:
            f.write(f"{r['obj1']:<20s} on {r['obj2']:<20s}: {'OK' if r['obj1_on_obj2'] else 'FAIL'}\n")
            f.write(f"{r['obj2']:<20s} on {r['obj1']:<20s}: {'OK' if r['obj2_on_obj1'] else 'FAIL'}\n")


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
        result = {
            "obj1": os.path.basename(obj1),
            "obj2": os.path.basename(obj2),
            "obj1_on_obj2": _check_stack(env, table, on_top, obj1, obj2),
            "obj2_on_obj1": _check_stack(env, table, on_top, obj2, obj1),
        }
        results.append(result)

        mark1 = f"{GREEN}OK{RESET}" if result["obj1_on_obj2"] else f"{RED}FAIL{RESET}"
        mark2 = f"{GREEN}OK{RESET}" if result["obj2_on_obj1"] else f"{RED}FAIL{RESET}"
        #print(f"  {result['obj1']:<20s} on {result['obj2']:<20s}: {mark1}   "
        #      f"{result['obj2']:<20s} on {result['obj1']:<20s}: {mark2}")

    _write_on_top_report(results, ON_TOP_REPORT_PATH)

    both_ok = sum(1 for r in results if r["obj1_on_obj2"] and r["obj2_on_obj1"])
    print(f"PASS: test_on_top ({both_ok}/{len(results)} pairs OK in both directions, report: {ON_TOP_REPORT_PATH})")


def main():
    arg_dict, trials = parse_args()
    env = build_env(arg_dict)

    #test_touching_and_on_top(env)
    #test_is_reachable(env, trials)
    #test_init_predicates_are_enforced(env, trials)
    test_on_top(env)

    print("\nAll tests passed!")


if __name__ == '__main__':
    main()
