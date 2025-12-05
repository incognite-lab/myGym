import argparse
import copy
import importlib.resources as pkg_resources
import json
import os
import random
import time
from typing import Callable, Dict, Any, List

import commentjson
import gymnasium as gym
import numpy as np

# --- Project-Specific Imports ---
from myGym.envs.gym_env import GymEnv
from myGym.envs.natural_language import NaturalLanguage
from myGym.utils.callbacksSB3 import (
    SaveOnBestTrainingRewardCallback,
    MultiPPOEvalCallback,
    PPOEvalCallback,
)
from myGym.stable_baselines_mygym.multi_ppo_SB3 import MultiPPOSB3
from myGym.stable_baselines_mygym.ppoSB3 import PPO as PPO_MYGYM
from myGym.stable_baselines_mygym.Subproc_vec_envSB3 import SubprocVecEnv

# --- Stable-Baselines3 Imports (Try/Except for robustness) ---
try:
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3 import A2C, SAC, TD3, PPO  # Import standard SB3 algos
except ImportError as e:
    print(f"Warning: Stable-Baselines3 components not fully available. Error: {e}")
    # Define dummy variables for models if import fails
    A2C, SAC, TD3, PPO = None, None, None, None

try:
    # Attempt import for the HER wrapper if needed
    from stable_baselines3.common.env_wrappers import HERGoalEnvWrapper
except ImportError:
    HERGoalEnvWrapper = None
    pass

# Suppress TensorFlow logging and optimize thread usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

# Global Variables
AVAILABLE_SIMULATION_ENGINES: List[str] = ["pybullet"]
AVAILABLE_TRAINING_FRAMEWORKS: List[str] = ["pytorch"]


def save_results(arg_dict: Dict[str, Any], model_name: str, env: gym.Env, model_logdir: str = None,
                 show: bool = False) -> None:
    """
    Prints a success message and final log directory.
    """
    logdir = model_logdir if model_logdir is not None else arg_dict["logdir"]
    print(f"Model log directory: {logdir}")
    print(f"Training with {arg_dict['steps']} timesteps succeeded!")


def configure_env(arg_dict: Dict[str, Any], model_logdir: str = None, for_train: bool = True) -> gym.Env:
    """
    Creates and configures the Gym environment based on argument dictionary.
    """
    # Consolidated environment arguments for GymEnv
    env_arguments = {
        "render_on": True,
        "visualize": arg_dict["visualize"],
        "workspace": arg_dict["workspace"],
        "robot": arg_dict["robot"],
        "robot_init_joint_poses": arg_dict["robot_init"],
        "robot_action": arg_dict["robot_action"],
        "max_velocity": arg_dict["max_velocity"],
        "max_force": arg_dict["max_force"],
        "task_type": arg_dict["task_type"],
        "action_repeat": arg_dict["action_repeat"],
        "task_objects": arg_dict["task_objects"],
        "observation": arg_dict["observation"],
        "framework": "SB3",
        "distractors": arg_dict["distractors"],
        "num_networks": arg_dict.get("num_networks", 1),
        "network_switcher": arg_dict.get("network_switcher", "gt"),
        "distance_type": arg_dict["distance_type"],
        "used_objects": arg_dict["used_objects"],
        "active_cameras": arg_dict["camera"],
        "color_dict": arg_dict.get("color_dict", {}),
        "visgym": arg_dict["visgym"],
        "reward": arg_dict["reward"],
        "logdir": arg_dict["logdir"],
        "vae_path": arg_dict["vae_path"],
        "yolact_path": arg_dict["yolact_path"],
        "yolact_config": arg_dict["yolact_config"],
        "natural_language": bool(arg_dict["natural_language"]),
        "training": for_train,
        "top_grasp": arg_dict["top_grasp"],
        "max_ep_steps": arg_dict["max_episode_steps"],
        "gui_on": arg_dict["gui"],
    }

    obs_space = "dict" if arg_dict["algo"] == "her" else "default"
    env = gym.make(arg_dict["env_name"], **env_arguments, obs_space=obs_space)

    # Set max episode steps for non-HER environments
    if arg_dict["algo"] != "her":
        env.spec.max_episode_steps = 512

    if for_train:
        if arg_dict["engine"] == "mujoco":
            # Mujoco logic uses VecMonitor or Monitor depending on multiprocessing
            if arg_dict["multiprocessing"]:
                env = VecMonitor(env, model_logdir)
            else:
                env = Monitor(env, model_logdir)
        elif arg_dict["engine"] == "pybullet" and not arg_dict["multiprocessing"]:
            # Pybullet non-multiprocessing needs info_keywords for save_results
            env = Monitor(env, filename=model_logdir, info_keywords=tuple('d'))

    # Wrap for HER if required and the wrapper is available
    if arg_dict["algo"] == "her":
        if HERGoalEnvWrapper:
            env = HERGoalEnvWrapper(env)
        else:
            print("Warning: HERGoalEnvWrapper not available, HER training may fail.")

    return env


def make_env(arg_dict: Dict[str, Any], rank: int, seed: int = 0, model_logdir: str = None) -> Callable:
    """
    Utility function for multiprocessed environments (SubprocVecEnv).
    """

    def _init():
        # Ensure the custom Gym environment is registered once
        if "Gym-v0" not in gym.registry:
            gym.register("Gym-v0", GymEnv)

        # Set unique seed for each environment process
        arg_dict["seed"] = seed + rank
        env = configure_env(arg_dict, for_train=True, model_logdir=model_logdir)
        env.reset()
        return env

    set_random_seed(seed)
    return _init


def configure_implemented_combos(env: gym.Env, model_logdir: str, arg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maps algorithm and training framework to the Stable-Baselines3 model class and initialization arguments.
    """
    algo_steps = arg_dict["algo_steps"]
    log_kwargs = {"verbose": 1, "tensorboard_log": model_logdir, "device": "cpu"}

    # Base configuration for standard SB3 models
    base_config = ('MlpPolicy', env)

    implemented_combos: Dict[str, Any] = {
        "ppo": {"pytorch": [PPO_MYGYM, base_config, {"n_steps": algo_steps, **log_kwargs}]},
        "sac": {"pytorch": [SAC, base_config, log_kwargs]},
        "td3": {"pytorch": [TD3, base_config, log_kwargs]},
        "a2c": {"pytorch": [A2C, base_config, {"n_steps": algo_steps, **log_kwargs}]},
        "multippo": {
            "pytorch": [
                MultiPPOSB3,
                base_config,
                {"n_steps": algo_steps, "n_models": arg_dict["num_networks"], **log_kwargs}
            ]
        }
    }
    return implemented_combos


def setup_decider_for_multi_ppo(model, env, vec_env, model_logdir: str, pretrained_model: str = None) -> None:
    """
    Handles the complex logic for ensuring the DeciderPolicy (used for network switching)
    is correctly initialized/loaded and attached to both the environment's reward
    object and the MultiPPO model instance.
    """
    try:
        from myGym.stable_baselines_mygym.decider import DeciderPolicy, flatten_obs_any
    except Exception:
        DeciderPolicy = None
        flatten_obs_any = None

    if DeciderPolicy is None:
        print(">>> WARNING: DeciderPolicy import failed; skipping decider setup.")
        return

    # determine the concrete environment (env0) for reward access
    env_for_attach = vec_env if vec_env is not None else env
    env0 = env_for_attach.envs[0] if hasattr(env_for_attach, "envs") and len(
        env_for_attach.envs) > 0 else env_for_attach

    if env0 is None:
        print(">>> WARNING: Could not find concrete environment (env0) to attach decider.")
        return

    reward_obj = getattr(env0, "reward", None) or getattr(getattr(env0, "unwrapped", None), "reward", None)

    if reward_obj is None:
        print(">>> WARNING: Could not find reward object to attach decider.")
        return

    # handle state reset if loading a pretrained model
    if pretrained_model:
        print(">>> Resetting Decider state after loading pretrained model...")
        if hasattr(model, "current_network_idx"):
            model.current_network_idx = None
        if hasattr(model, "lock_until_step"):
            model.lock_until_step = 0
        if hasattr(model, "step_counter"):
            model.step_counter = 0

    decider_log_path = os.path.join(model_logdir, "decider_log.tsv")
    # ensure decider is initialized and attached to the reward object
    if not hasattr(reward_obj, "decider_model") or getattr(reward_obj, "decider_model", None) is None:
        # infer dimensions if decider is not present
        obs_dim = getattr(reward_obj, "decider_obs_dim", 1)
        if obs_dim == 1 and flatten_obs_any:
            try:
                sample_obs = env0.observation_space.sample() if hasattr(env0,
                                                                        "observation_space") else env0.get_observation()
                flat = flatten_obs_any(sample_obs)
                obs_dim = int(flat.shape[0]) if flat is not None else 1
            except Exception:
                pass  # keep default 1

        network_names = getattr(reward_obj, "network_names", None)
        num_nets = int(len(network_names)) if network_names else int(getattr(env0, "num_networks", 1))

        reward_obj.decider_model = DeciderPolicy(obs_dim=obs_dim, num_networks=num_nets, log_path=decider_log_path)
        print(f">>> Decider model CREATED (obs_dim={obs_dim}, num_networks={num_nets}) and attached to reward.")
    else:
        if hasattr(reward_obj.decider_model, "set_log_path"):
            reward_obj.decider_model.set_log_path(decider_log_path)
        print(">>> Decider model already present on reward — keeping it.")

    # attach reward's decider model to the MultiPPO model instance
    if hasattr(reward_obj, "decider_model") and reward_obj.decider_model is not None:
        try:
            setattr(model, "decider", reward_obj.decider_model)
            print(">>> Attached reward.decider_model to model.decider.")
        except Exception as e:
            print(f">>> Warning: Failed to attach decider to model: {repr(e)}")


def task_objects_replacement(task_objects_new: List[str], task_objects_old: List[Dict[str, Any]], task_type: str) -> \
        List[Dict[str, Any]]:
    """
    Updates the object names in the task configuration structure.

    If task_objects is given as a parameter, this method converts string into a proper format
    depending on task_type (e.g., 'goal' for reach, 'init' for others).
    """
    if len(task_objects_new) > len(task_objects_old):
        msg = "More objects given than there are subtasks in the config."
        raise ValueError(msg)

    # Determine which part of the task config to update
    dest_key = "goal" if task_type == "reach" else "init"

    ret = copy.deepcopy(task_objects_old)
    for i, new_obj_name in enumerate(task_objects_new):
        ret[i][dest_key]["obj_name"] = new_obj_name
    return ret


def process_natural_language_command(cmd: str, env: gym.Env,
                                     output_relative_path: str = os.path.join("envs", "examples",
                                                                              "natural_language.txt")) -> None:
    """
    Generates natural language output (description or new tasks) and saves it to a file.
    """
    env.reset()
    nl = NaturalLanguage(env)
    output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), output_relative_path)

    if cmd == "description":
        output = nl.generate_task_description()
        print(f"Generating task description and saving to {output_path}")
    elif cmd == "new_tasks":
        output = "\n".join(nl.generate_new_tasks())
        print(f"Generating new tasks and saving to {output_path}")
    else:
        raise ValueError(f"Unknown natural language command: {cmd}")

    with open(output_path, "w") as file:
        file.write(output)


def automatic_argument_assignment(arg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Infers missing arguments (num_networks, reward, logdir, algo_steps) from the task_type string.
    """
    task_type_str = arg_dict.get("task_type")

    if task_type_str and isinstance(task_type_str, str):
        # Infer properties from the number of networks/subtasks in task_type string
        num_networks = len(task_type_str)
        logdir_base = f"./trained_models/{arg_dict['robot']}/{arg_dict['task_type']}"

        arg_dict["num_networks"] = num_networks
        arg_dict["reward"] = arg_dict["task_type"]
        arg_dict["logdir"] = logdir_base
        arg_dict["algo_steps"] = arg_dict["max_episode_steps"]

        print(f"Inferred number of networks: {num_networks}")
        print(f"Inferred reward type: {arg_dict['reward']}")
        print(f"Inferred log directory: {arg_dict['logdir']}")
        print(f"Inferred algorithm steps: {arg_dict['algo_steps']}")
    else:
        # default if task_type is missing or invalid
        arg_dict["num_networks"] = 1
        arg_dict["reward"] = "None"

    return arg_dict


def train(env: gym.Env, implemented_combos: Dict[str, Any], model_logdir: str, arg_dict: Dict[str, Any],
          pretrained_model: str = None):
    """
    Initializes and runs the Stable-Baselines3 training loop.
    """
    model_name = f"{arg_dict['algo']}_{arg_dict['steps']}"
    steps_trained = 0

    # handle training from scratch vs. loading a checkpoint
    if not pretrained_model:
        # training from scratch: save config and initialize steps counter
        conf_pth = os.path.join(model_logdir, "train.json")
        with open(conf_pth, "w") as f:
            json.dump(arg_dict, f, indent=4)
        with open(os.path.join(model_logdir, "trained_steps.txt"), "a+") as f:
            f.write(f"model {model_name} has been saved at steps:\n")
    else:
        # loading checkpoint: read trained steps and update logdir
        trained_steps_file = os.path.join(pretrained_model, "trained_steps.txt")
        if os.path.exists(trained_steps_file):
            with open(trained_steps_file, "r") as f:
                lines = f.readlines()
                # assuming the last line holds the last saved step count
                steps_trained = int(lines[-1].strip().split()[-1])
        model_logdir = pretrained_model

    # model initialization
    algo_config = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]]
    ModelClass, model_args_tuple, model_kwargs = algo_config

    # set seed if provided
    seed = arg_dict.get("seed", None)
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        model_kwargs["seed"] = seed

    vec_env = None  # initialize VecEnv variable

    if pretrained_model:
        # load the model
        if not os.path.isabs(pretrained_model):
            # resolve relative paths relative to myGym package
            pretrained_model = os.path.join(pkg_resources.files("myGym"), pretrained_model)

        # need to re-create the vectorized environment for loading
        if not arg_dict["multiprocessing"]:
            vec_env = DummyVecEnv([lambda: env])
        else:
            vec_env = env  # the env passed here is already a SubprocVecEnv

        model = ModelClass.load(pretrained_model, vec_env, device="cpu")
    else:
        # create a new model
        model = ModelClass(*model_args_tuple, **model_kwargs)

    # decider/MultiPPO setup (must run after model creation/loading)
    if arg_dict["algo"] == "multippo" and arg_dict.get("network_switcher") == "decider":
        try:
            setup_decider_for_multi_ppo(model, env, vec_env, model_logdir, pretrained_model)
        except Exception as e:
            print(f">>> CRITICAL ERROR during Decider setup: {repr(e)}")

    # GAIL logic (simplified, assuming support is available)
    if arg_dict["algo"] == "gail":
        print("GAIL training is configured but not fully implemented in this block.")

    # callbacks and training
    start_time = time.time()
    callbacks_list: List[BaseCallback] = []

    # auto-save callback
    auto_save_callback = SaveOnBestTrainingRewardCallback(
        check_freq=1024,
        logdir=model_logdir,
        env=env,
        engine=arg_dict["engine"],
        multiprocessing=arg_dict["multiprocessing"],
        save_model_every_steps=arg_dict["eval_freq"],
        starting_steps=steps_trained,
        algo=arg_dict["algo"],
    )
    callbacks_list.append(auto_save_callback)

    # evaluation callback
    if arg_dict["eval_freq"]:
        NUM_CPU = int(arg_dict["multiprocessing"]) if arg_dict["multiprocessing"] else 1

        EvalCallbackClass = MultiPPOEvalCallback if arg_dict["algo"] == "multippo" else PPOEvalCallback

        # attach decider to eval env
        if arg_dict.get("network_switcher") == "decider":
            try:
                setup_decider_for_multi_ppo(model, env, None, pretrained_model)
            except Exception as e:
                print(">>> WARNING: Failed to attach decider to eval_env:", e)

        eval_callback = EvalCallbackClass(
            env,  # using the primary environment/vec_env for evaluation
            log_path=model_logdir,
            eval_freq=arg_dict["eval_freq"],
            algo_steps=arg_dict["algo_steps"],
            n_eval_episodes=arg_dict["eval_episodes"],
            record=arg_dict["record"],
            camera_id=arg_dict["camera"],
            num_cpu=NUM_CPU,
            starting_steps=steps_trained,
        )

        callbacks_list.append(eval_callback)

    print("--- Training Started ---")

    total_steps_to_train = arg_dict["steps"]
    model.learn(total_timesteps=total_steps_to_train, callback=callbacks_list)
    print("--- Training Ended ---")

    # save final model and results
    final_steps = steps_trained + model.num_timesteps
    print(f'Saving final model at steps {final_steps} to {model_logdir}')
    model.save(model_logdir, steps=final_steps)

    env.close()

    print(f"Training time: {time.time() - start_time:.2f} s")
    print(f"Training steps: {model.num_timesteps}")

    if arg_dict["engine"] == "pybullet":
        save_results(arg_dict, model_name, env, model_logdir)

    return model


def get_parser() -> argparse.ArgumentParser:
    """
    Defines and returns the argument parser for the training script.
    """
    parser = argparse.ArgumentParser(description="MyGym RL Training Script")

    # --- Configuration ---
    parser.add_argument("-cfg", "--config", type=str, default="./configs/train_AGM_RDDL.json",
                        help="Path to config file.")

    # --- Environment Args ---
    env_group = parser.add_argument_group("Environment and Simulation")
    env_group.add_argument("-n", "--env_name", type=str, help="Environment name.")
    env_group.add_argument("-ws", "--workspace", type=str, help="Workspace name.")
    env_group.add_argument("-p", "--engine", type=str, help="Simulation engine name.")
    env_group.add_argument("-sd", "--seed", type=int, default=1, help="Seed number for reproducibility.")
    env_group.add_argument("-d", "--render", type=str, help="Rendering type: opengl, opencv.")
    env_group.add_argument("-c", "--camera", type=int, help="Number of cameras for rendering/recording.")
    env_group.add_argument("-vi", "--visualize", type=int, help="Visualize camera render and vision (1/0).")
    env_group.add_argument("-vg", "--visgym", type=int, help="Visualize gym background (1/0).")
    env_group.add_argument("-g", "--gui", type=int, help="Use GUI (1/0).")

    # --- Robot Args ---
    robot_group = parser.add_argument_group("Robot Configuration")
    robot_group.add_argument("-b", "--robot", default=["kuka", "panda"], nargs='*',
                             help="Robot to train (e.g., kuka, panda).")
    robot_group.add_argument("-bi", "--robot_init", nargs="*", type=float,
                             help="Initial robot's end-effector position.")
    robot_group.add_argument("-ba", "--robot_action", type=str, help="Robot's action control (step, absolute, joints).")
    robot_group.add_argument("-mv", "--max_velocity", type=float, help="Maximum velocity of robotic arm.")
    robot_group.add_argument("-mf", "--max_force", type=float, help="Maximum force of robotic arm.")
    robot_group.add_argument("-ar", "--action_repeat", type=int, help="Substeps of simulation without action from env.")
    robot_group.add_argument("-no", "--observed_links_num", type=int,
                             help="Number of robot links in observation space.")

    # --- Task & Reward Args ---
    task_group = parser.add_argument_group("Task and Reward Configuration")
    task_group.add_argument("-tt", "--task_type", type=str, help="Type of task (e.g., reach, push).")
    task_group.add_argument("-to", "--task_objects", nargs="*", type=str, help="Object(s) to manipulate.")
    task_group.add_argument("-u", "--used_objects", nargs="*", type=str, help="Extra objects for scene randomization.")
    task_group.add_argument("-re", "--reward", type=str, help="Defines how to compute the reward.")
    task_group.add_argument("-dt", "--distance_type", type=str, help="Type of distance metrics: euclidean, manhattan.")

    # --- Distractor Args ---
    distractor_group = parser.add_argument_group("Distractor Configuration")
    distractor_group.add_argument("-di", "--distractors", type=str, help="Object to evade (distractor).")
    distractor_group.add_argument("-dm", "--distractor_moveable", type=int, help="Can distractor move (0/1).")
    distractor_group.add_argument("-ds", "--distractor_constant_speed", type=int,
                                  help="Is speed of distractor constant (0/1).")
    distractor_group.add_argument("-dd", "--distractor_movement_dimensions", type=int,
                                  help="Movement dimensions (1/2/3).")
    distractor_group.add_argument("-de", "--distractor_movement_endpoints", nargs="*", type=float,
                                  help="Movement endpoints (2 coordinates).")

    # --- Training Args ---
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument("-w", "--train_framework", type=str, help="Training framework: {tensorflow, pytorch}.")
    train_group.add_argument("-a", "--algo", type=str, help="The learning algorithm (e.g., ppo, her).")
    train_group.add_argument("-s", "--steps", type=int, help="The total number of steps to train.")
    train_group.add_argument("-ms", "--max_episode_steps", type=int, help="The maximum number of steps per episode.")
    train_group.add_argument("-ma", "--algo_steps", type=int,
                             help="The number of steps per algo training iteration (PPO, A2C).")
    train_group.add_argument("-i", "--multiprocessing", type=int,
                             help="Number of vectorized environments for multiprocessing (0 or >1).")

    # --- Evaluation & Logging Args ---
    log_group = parser.add_argument_group("Evaluation and Logging")
    log_group.add_argument("-ef", "--eval_freq", type=int, help="Evaluate the agent every eval_freq steps.")
    log_group.add_argument("-e", "--eval_episodes", type=int, help="Number of episodes to evaluate performance.")
    log_group.add_argument("-l", "--logdir", type=str, help="Where to save results of training and trained models.")
    log_group.add_argument("-r", "--record", type=int, help="0: don't record, 1: gif, 2: video.")

    # --- Paths & Pretrained Models ---
    path_group = parser.add_argument_group("Model and VAE Paths")
    path_group.add_argument("-m", "--model_path", type=str, help="Path to the trained model to test.")
    path_group.add_argument("-vp", "--vae_path", type=str, help="Path to a trained VAE in 2dvu reward type.")
    path_group.add_argument("-yp", "--yolact_path", type=str, help="Path to a trained Yolact in 3dvu reward type.")
    path_group.add_argument("-yc", "--yolact_config", type=str, help="Yolact config path or name.")
    path_group.add_argument('-ptm', "--pretrained_model", type=str, help="Path to a model for continued training.")

    # --- Natural Language (NL) ---
    nl_group = parser.add_argument_group("Natural Language Output")
    nl_group.add_argument("-nl", "--natural_language", type=str, default="",
                          help="If passed, generate 'description' or 'new_tasks' instead of training.")

    return parser


def get_arguments(parser: argparse.ArgumentParser) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Parses command-line arguments and loads/merges them with arguments from the config file.
    Command-line arguments override config file arguments.
    """
    args = parser.parse_args()

    # load arguments from the config file
    with open(args.config, "r") as f:
        arg_dict = commentjson.load(f)

    # convert initial lists in config to single values if applicable (clean up)
    for key, value in arg_dict.items():
        if value is not None and key != "config":
            if key in ["robot_init", "end_effector_orn"]:
                # convert list of strings/ints/floats to list of floats
                arg_dict[key] = [float(item) for item in value]
            elif isinstance(value, list) and len(value) == 1 and key != "task_objects":
                arg_dict[key] = value[0]

    # merge/override with command-line arguments
    command_line_overrides: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if value is not None and key != "config":
            is_new_or_overridden = (key not in arg_dict or arg_dict[key] is None or value != parser.get_default(key))

            if is_new_or_overridden:
                if key == "task_objects":
                    # special handling for task_objects to update the complex structure
                    if key in arg_dict and arg_dict[key]:
                        arg_dict[key] = task_objects_replacement(value, arg_dict[key], arg_dict["task_type"])

                    # store the raw command line value for display/tracking
                    command_line_overrides[key] = value[0] if isinstance(value, list) and len(value) == 1 else value

                elif isinstance(value, list) and len(value) <= 1 and key not in ["robot"]:
                    # clean up single-element list from CLI args
                    arg_dict[key] = value[0]
                    command_line_overrides[key] = value[0]
                else:
                    arg_dict[key] = value
                    command_line_overrides[key] = value

    return arg_dict, command_line_overrides


def main():
    """
    Main entry point for the training script.
    """
    parser = get_parser()
    arg_dict, _ = get_arguments(parser)

    # finalize argument setup
    arg_dict["top_grasp"] = False

    # input validation and automatic assignment
    if arg_dict["engine"] not in AVAILABLE_SIMULATION_ENGINES:
        print(f"Invalid simulation engine. Valid arguments: --engine {AVAILABLE_SIMULATION_ENGINES}.")
        return

    if not os.path.isabs(arg_dict.get("logdir", "")):
        arg_dict = automatic_argument_assignment(arg_dict)

    # Natural Language Processing (exit if command is given)
    nl_cmd = arg_dict.get("natural_language")
    if nl_cmd:
        # Need a temporary environment just for NL generation
        temp_env = configure_env(arg_dict, for_train=False)
        process_natural_language_command(nl_cmd, temp_env)
        temp_env.close()
        return  # exit the program after NL generation

    # ensure logdir is an absolute path (or relative to execution)
    if not os.path.isabs(arg_dict["logdir"]):
        arg_dict["logdir"] = os.path.join(os.getcwd(), arg_dict["logdir"])

    os.makedirs(arg_dict["logdir"], exist_ok=True)

    # create unique model log directory
    model_logdir_base = os.path.join(arg_dict["logdir"], f"{arg_dict['robot_action']}_{arg_dict['algo']}")
    model_logdir = model_logdir_base

    if not arg_dict["pretrained_model"]:
        # training from scratch: find a unique path (model_logdir_base_1, _2, etc.)
        add = 1
        while True:
            try:
                os.makedirs(model_logdir, exist_ok=False)
                break
            except FileExistsError:
                model_logdir = f"{model_logdir_base}_{add}"
                add += 1
    else:
        # renewing training: logdir is the parent of the checkpoint
        model_logdir = os.path.dirname(arg_dict["pretrained_model"])

    # environment setup (vectorized or single)
    if arg_dict["multiprocessing"] and int(arg_dict["multiprocessing"]) > 0:
        NUM_CPU = int(arg_dict["multiprocessing"])
        print(f"Initializing {NUM_CPU} vectorized environments...")
        env = SubprocVecEnv([make_env(arg_dict, i, model_logdir=model_logdir) for i in range(NUM_CPU)])
        # VecMonitor is applied inside make_env for SubprocVecEnv, but we need it here for the wrapper:
        # env = VecMonitor(env, model_logdir)
    else:
        env = configure_env(arg_dict, model_logdir, for_train=True)

    # training execution
    implemented_combos = configure_implemented_combos(env, model_logdir, arg_dict)
    train(env, implemented_combos, model_logdir, arg_dict, arg_dict["pretrained_model"])


if __name__ == "__main__":
    main()
