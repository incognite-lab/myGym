import argparse
import json
import os
import re
from math import ceil as ceiling
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# robust JSON loading
def load_concatenated_json(path: str):
    """
    Loads a file that may contain:
      - a single JSON object/array, OR
      - multiple JSON objects appended one after another.
    Returns:
      - obj if single
      - list[obj] if multiple
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        s = f.read().strip()
    if not s:
        return None

    dec = json.JSONDecoder()
    i = 0
    objs = []
    n = len(s)

    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break
        obj, j = dec.raw_decode(s, i)
        objs.append(obj)
        i = j

    return objs[0] if len(objs) == 1 else objs


_float_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_scalar(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v)
    if isinstance(v, str):
        m = _float_re.search(v)
        return float(m.group(0)) if m else None
    return None


def parse_vector(v: Any) -> Optional[List[float]]:
    """
    Accepts:
      - list/tuple of numbers
      - string like "[1. 2. 3.]" or "[-0.1  0.2  0.3]"
    Returns list[float] or None.
    """
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        try:
            return [float(x) for x in v]
        except Exception:
            return None
    if isinstance(v, str):
        nums = _float_re.findall(v)
        if not nums:
            return None
        return [float(x) for x in nums]
    return None


def read_eval_timeseries(eval_path: str) -> List[Dict[str, Any]]:
    """
    Handles:
      A) concatenated objects like:
         {"evaluation_after_123_steps": {...}}{"evaluation_after_456_steps": {...}}...
      B) a list of such dicts
      C) a single dict with many evaluation_after_* keys
    Returns list of records with a normalized 'training_steps' int key.
    """
    raw = load_concatenated_json(eval_path)
    if raw is None:
        return []

    objs: List[Any]
    if isinstance(raw, list):
        objs = raw
    else:
        objs = [raw]

    records: List[Dict[str, Any]] = []

    for obj in objs:
        if isinstance(obj, dict):
            # case C: one dict with many evaluation_after_* keys
            if any(isinstance(k, str) and "evaluation_after_" in k for k in obj.keys()) and len(obj) > 1:
                for k, inner in obj.items():
                    if not (isinstance(k, str) and "evaluation_after_" in k and isinstance(inner, dict)):
                        continue
                    m = re.search(r"evaluation_after_(\d+)_steps", k)
                    step = int(m.group(1)) if m else None
                    rec = {"training_steps": step}
                    rec.update(inner)
                    records.append(rec)
                continue

            # case A: each obj is {"evaluation_after_X_steps": {...}} (len==1)
            if len(obj) == 1:
                k = next(iter(obj.keys()))
                inner = obj[k]
                if isinstance(k, str) and "evaluation_after_" in k and isinstance(inner, dict):
                    m = re.search(r"evaluation_after_(\d+)_steps", k)
                    step = int(m.group(1)) if m else None
                    rec = {"training_steps": step}
                    rec.update(inner)
                    records.append(rec)
                    continue

            # fallback: already a record dict
            if "training_steps" in obj:
                records.append(obj)

    # drop records without step, sort by step
    records = [r for r in records if r.get("training_steps") is not None]
    records.sort(key=lambda r: int(r["training_steps"]))
    return records


# Color mapping for algorithms
color_map: Dict[str, str] = {
    "ppo": "red",
    "ppo2": "green",
    "multippo": "aquamarine",
    "acktr": "yellow",
    "multiacktr": "magenta",
    "sac": "salmon",
    "ddpg": "blue",
    "a2c": "grey",
    "acer": "brown",
    "trpo": "gold",
    "multippo2": "limegreen"
}

# Action dictionary
dict_acts: Dict[str, str] = {
    "A": "approach", "G": "grasp", "M": "move", "D": "drop",
    "W": "withdraw", "R": "rotate", "T": "transform",
    "F": "find", "r": "reach", "L": "leave"
}

# Color mapping for actions
color_map_acts: Dict[str, str] = {
    "approach": "#339dff",  # muted blue
    "withdraw": "#faa22e",  # muted orange
    "grasp": "#10e600",  # muted green
    "drop": "#f1160e",  # muted red
    "move": "#a838ff",  # muted purple
    "rotate": "#745339",  # muted brown
    "transform": "#f787d3",  # muted pink
    "follow": "#99acb8",  # muted gray
    "reach": "#fff35c",  # muted yellow
    "find": "#339dff",  # muted blue
    "leave": "#faa22e"  # muted orange
}


def cut_before_last_slash(logdir: str) -> str:
    """Cut all strings prior to and including the last '/' in the given string."""
    parts = logdir.replace("\\", "/").rsplit('/', 1)
    return parts[1] if len(parts) > 1 else logdir


def atoi(text: str) -> Any:
    """Convert text to integer if it's a digit, otherwise return the text."""
    return int(text) if text.isdigit() else text


def natural_keys(text: str) -> List[Any]:
    """Split text into natural sort keys."""
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def cfg_string2dict(cfg_raw: str) -> Dict[str, str]:
    """Convert a configuration string to a dictionary."""
    return {key: value for key, _, value, _ in re.findall(
        r"(\w+\s?)(=\s?)([^=]+)( (?=\w+\s?=)|$)", cfg_raw)}


def dict2cfg_string(dictionary: Dict[str, str], separator: str = "\n", assigner: str = "=") -> str:
    """Convert a dictionary into a configuration string."""
    return separator.join([f"{k}{assigner}{v}" for k, v in dictionary.items()])


def multi_dict_diff_by_line(dict_list: List[Dict[str, Any]]) -> Tuple[Dict[str, List[Any]], Dict[str, Any]]:
    """Compare multiple dictionaries line-by-line."""
    different_items, same_items = {}, {}
    for record in zip(*[d.items() for d in dict_list]):
        vals = [v for _, v in record]
        keys = [k for k, _ in record]
        if not all(k == keys[0] for k in keys):
            raise ValueError("Inconsistent keys in the configurations.")
        if all(v == vals[0] for v in vals):
            same_items[keys[0]] = vals[0]
        else:
            different_items[keys[0]] = vals
    return different_items, same_items


def multi_dict_diff_scary(dict_list: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[bool]]:
    """Compare a list of dictionaries and return differences and similarities."""
    all_vals = [[d.get(k) for d in dict_list] for k in dict_list[0].keys()]
    all_same = [all(v == line[0] for v in line) for line in all_vals]
    diff_dict = {record[0][0]: (record[0][1] if all_same[i] else [v for _, v in record])
                 for i, record in enumerate(zip(*[d.items() for d in dict_list]))}
    return diff_dict, all_same


def multi_dict_diff_by_key(dict_list: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compare dictionaries by keys and return differences and similarities."""
    diff_dict, all_same = multi_dict_diff_scary(dict_list)
    return {k: v for i, (k, v) in enumerate(diff_dict.items()) if not all_same[i]}, \
        {k: v for i, (k, v) in enumerate(diff_dict.items()) if all_same[i]}


def get_arguments() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth",
                        default="C:/Users/xosti/sofia/PycharmProjects/myGym/myGym/trained_models/tiago_dual/AGM/joints_gripper_multippo_6",
                        type=str)
    parser.add_argument("--robot", default=["kuka", "panda"], nargs='*', type=str)
    parser.add_argument("--algo", default=["multiacktr", "multippo2", "ppo2", "ppo", "acktr", "sac", "ddpg",
                                           "a2c", "acer", "trpo", "multippo"], nargs='*', type=str)
    return parser.parse_args()


def legend_without_duplicate_labels(ax: plt.Axes) -> None:
    """Create a legend without duplicate labels."""
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    if unique:
        ax.legend(*zip(*unique), fontsize="12", loc="center right")


def ax_set(ax: plt.Axes, title: str, y_axis: str) -> None:
    """Set title and labels for the given axis."""
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Training steps', fontsize=12)
    ax.set_ylabel(y_axis, fontsize=12)
    legend_without_duplicate_labels(ax)


def ax_plot(ax: plt.Axes, steps: np.ndarray, data: np.ndarray, color: str, label: str) -> None:
    """Plot data on the given axis."""
    ax.plot(steps, data, color=color, linestyle='solid', linewidth=1.5, marker='o',
            markerfacecolor=color, markersize=2, label=label)


def ax_fill(ax: plt.Axes, steps: np.ndarray, meanvalue: np.ndarray, data: List[np.ndarray],
            index: List[int], color: str) -> None:
    """Fill the area between the lines on the given axis."""
    if len(index) <= 1:
        return
    ax.fill_between(
        steps,
        meanvalue - np.std(np.take(data, index, 0), 0),
        meanvalue + np.std(np.take(data, index, 0), 0),
        color=color,
        alpha=0.15
    )


def plot_gt_vs_decider_bar(exp_dir: str,
                           acts: List[str],
                           out_dir: str,
                           cfg_raw: Dict[str, Any]) -> None:
    gt_file = os.path.join(exp_dir, "gt_vs_decider.txt")
    if not os.path.isfile(gt_file):
        print(f"No gt_vs_decider.txt in {exp_dir}, skipping GT vs Decider plot.")
        return

    try:
        data = np.loadtxt(gt_file, delimiter=",", dtype=int)
    except Exception as e:
        print(f"Could not load {gt_file}: {e}")
        return

    if data.ndim == 1:
        data = data[None, :]

    # columns: episode, step, gt, dec
    gt = data[:, 2]
    dec = data[:, 3]

    num_networks = len(acts)
    total = len(gt)
    if total == 0:
        print("gt_vs_decider.txt is empty.")
        return

    gt_counts = np.array([(gt == i).sum() for i in range(num_networks)], dtype=float)
    dec_counts = np.array([(dec == i).sum() for i in range(num_networks)], dtype=float)

    gt_pct = gt_counts / total * 100.0
    dec_pct = dec_counts / total * 100.0

    x = np.arange(num_networks)
    width = 0.35

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.bar(x - width / 2, gt_pct, width, label="GT", color="lightgray", edgecolor="black")

    bar_colors = [color_map_acts.get(a, "black") for a in acts]
    bars_dec = ax.bar(x + width / 2, dec_pct, width, label="Decider", color=bar_colors, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(acts, fontsize=12)
    ax.set_ylabel("Selection frequency (%)")
    ax.set_ylim(0, 100)

    ax.set_title(f"GT vs Decider network choices\nalgo: {cfg_raw.get('algo')}, task: {cfg_raw.get('task_type')}")

    for i, b in enumerate(bars_dec):
        diff = dec_pct[i] - gt_pct[i]
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 1.0, f"{diff:+.1f}",
                ha="center", va="bottom", fontsize=10)

    ax.legend()
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cut_before_last_slash(cfg_raw.get('logdir', 'exp'))}_gt_vs_decider.png")
    fig.savefig(out_path, dpi=150)
    print(f"GT vs Decider figure saved to {out_path}")


def main() -> None:
    global color_map_acts
    args = get_arguments()

    out_dir = os.path.join(".", "trained_models")
    os.makedirs(out_dir, exist_ok=True)

    # Collect experiment dirs (subfolders with train.json + evaluation_results.json), and allow single-folder mode.
    if not os.path.isdir(args.pth):
        print(f"No such path: {args.pth}")
        return

    subdirs = next(os.walk(str(args.pth)))[1]

    def has_required_files(path: str) -> bool:
        return os.path.isfile(os.path.join(path, "train.json")) and os.path.isfile(
            os.path.join(path, "evaluation_results.json"))

    experiment_dirs: List[str] = []
    for d in sorted(subdirs, key=natural_keys):
        full = os.path.join(args.pth, d)
        if has_required_files(full):
            experiment_dirs.append(full)

    # Single-folder mode
    if has_required_files(args.pth) and args.pth not in experiment_dirs:
        experiment_dirs.append(args.pth)

    if not experiment_dirs:
        print("No experiments found (need train.json + evaluation_results.json).")
        return

    plt.rcParams.update({'font.size': 10})

    configs: List[Dict[str, Any]] = []

    # per-experiment arrays (each item is np.ndarray)
    steps_list: List[np.ndarray] = []
    success_list: List[np.ndarray] = []
    mean_steps_list: List[np.ndarray] = []
    mean_reward_list: List[np.ndarray] = []
    std_reward_list: List[np.ndarray] = []
    mean_dist_err_list: List[np.ndarray] = []
    mean_subgoals_finished_list: List[np.ndarray] = []

    # each item is np.ndarray shape (num_actions, T) with percentages; can be None if missing
    mean_subgoals_steps_list: List[Optional[np.ndarray]] = []
    acts_list: List[List[str]] = []

    min_T = None
    max_steps_cfg = None
    last_cfg = None

    for exp_dir in experiment_dirs:
        folder_name = os.path.basename(exp_dir.rstrip("/\\"))
        try:
            eval_path = os.path.join(exp_dir, "evaluation_results.json")
            records = read_eval_timeseries(eval_path)
            if not records:
                print(f"0 datapoints in folder (no eval records parsed): {folder_name}")
                continue

            # read config
            with open(os.path.join(exp_dir, "train.json"), "r", encoding="utf-8", errors="replace") as f:
                cfg_raw = yaml.full_load(f)
            configs.append(cfg_raw)
            last_cfg = cfg_raw
            max_steps_cfg = int(cfg_raw.get("max_episode_steps", 512))

            task = cfg_raw.get("task_type", "")
            acts = [dict_acts[c] for c in str(task) if c in dict_acts]
            acts_list.append(acts)

            # build numeric arrays
            steps = np.array([int(r["training_steps"]) for r in records], dtype=np.int64)

            success = np.array([parse_scalar(r.get("success_rate")) for r in records], dtype=np.float32)
            mean_steps = np.array([parse_scalar(r.get("mean_steps_num")) for r in records], dtype=np.float32)
            mean_reward = np.array([parse_scalar(r.get("mean_reward")) for r in records], dtype=np.float32)
            std_reward = np.array([parse_scalar(r.get("std_reward")) for r in records], dtype=np.float32)
            mean_dist_err = np.array([parse_scalar(r.get("mean_distance_error")) for r in records], dtype=np.float32)
            mean_subgoals_finished = np.array([parse_scalar(r.get("mean subgoals finished")) for r in records],
                                              dtype=np.float32)

            # replace Nones (if any) with nan
            for arr in (success, mean_steps, mean_reward, std_reward, mean_dist_err, mean_subgoals_finished):
                # arr is float32 with possible nan already; keep as is
                pass

            # subgoal steps (vector per record)
            sub_steps_vecs = [parse_vector(r.get("mean subgoal steps")) for r in records]
            if any(v is None for v in sub_steps_vecs):
                sub_steps = None
            else:
                sub_steps = np.array(sub_steps_vecs, dtype=np.float32)  # (T, K)
                # normalize to % of max episode steps
                sub_steps = (sub_steps / float(max_steps_cfg)) * 100.0
                sub_steps = sub_steps.T  # (K, T)
                # align K with acts (sometimes logs include extra zeros)
                K = min(sub_steps.shape[0], len(acts))
                sub_steps = sub_steps[:K, :]

            steps_list.append(steps)
            success_list.append(success)
            mean_steps_list.append(mean_steps)
            mean_reward_list.append(mean_reward)
            std_reward_list.append(std_reward)
            mean_dist_err_list.append(mean_dist_err)
            mean_subgoals_finished_list.append(mean_subgoals_finished)
            mean_subgoals_steps_list.append(sub_steps)

            min_T = len(steps) if min_T is None else min(min_T, len(steps))
            print(f"{len(steps)} datapoints in folder: {folder_name}")

        except Exception as e:
            print(f"Error processing {folder_name}: {e}")

    if not steps_list:
        print("No valid data extracted.")
        return

    # truncate all to the same length
    if min_T is None or min_T <= 0:
        print("No valid datapoints after parsing.")
        return

    steps_list = [s[:min_T] for s in steps_list]
    success_list = [s[:min_T] for s in success_list]
    mean_steps_list = [s[:min_T] for s in mean_steps_list]
    mean_reward_list = [s[:min_T] for s in mean_reward_list]
    std_reward_list = [s[:min_T] for s in std_reward_list]
    mean_dist_err_list = [s[:min_T] for s in mean_dist_err_list]
    mean_subgoals_finished_list = [s[:min_T] for s in mean_subgoals_finished_list]
    mean_subgoals_steps_list = [(m[:, :min_T] if m is not None else None) for m in mean_subgoals_steps_list]

    # Get differences between configs
    diff, _same = multi_dict_diff_by_key(configs)
    plot_num = len(set(diff['algo'])) if 'algo' in diff else 1
    if plot_num == 2:
        plot_num = 3

    # Plotting
    fig1 = plt.figure(1, figsize=(18, 12))
    ax1, ax2, ax3, ax4, ax5 = [fig1.add_subplot(3, 2, i) for i in range(1, 6)]

    fig2 = plt.figure(2, figsize=(13 * ceiling(plot_num / 2), 9 * ceiling(plot_num / 2)))

    counter = 0

    # grouping by algo if present
    if 'algo' in diff and len(args.algo) > 1:
        index = [[] for _ in range(len(args.algo))]
        for i, algo in enumerate(args.algo):
            for j, diffalgo in enumerate(diff['algo']):
                if algo == diffalgo:
                    index[i].append(j)

            if not index[i]:
                continue

            steps = steps_list[index[i][0]]

            # mean + std bands
            mean_success = np.nanmean(np.take(success_list, index[i], 0), 0)
            ax_plot(ax1, steps, mean_success, color_map.get(algo, "black"), algo)
            ax_fill(ax1, steps, mean_success, success_list, index[i], color_map.get(algo, "black"))

            mean_rew = np.nanmean(np.take(mean_reward_list, index[i], 0), 0)
            ax_plot(ax2, steps, mean_rew, color_map.get(algo, "black"), algo)
            ax_fill(ax2, steps, mean_rew, mean_reward_list, index[i], color_map.get(algo, "black"))

            mean_std = np.nanmean(np.take(std_reward_list, index[i], 0), 0)
            ax2.plot(steps, mean_std, alpha=0.8, color=color_map.get(algo, "black"),
                     linestyle='dotted', linewidth=1.5, marker='o',
                     markerfacecolor=color_map.get(algo, "black"), markersize=2, label=f"{algo} std")

            mean_steps = np.nanmean(np.take(mean_steps_list, index[i], 0), 0)
            ax_plot(ax3, steps, mean_steps, color_map.get(algo, "black"), algo)
            ax_fill(ax3, steps, mean_steps, mean_steps_list, index[i], color_map.get(algo, "black"))

            mean_err = np.nanmean(np.take(mean_dist_err_list, index[i], 0), 0)
            ax_plot(ax4, steps, mean_err, color_map.get(algo, "black"), algo)
            ax_fill(ax4, steps, mean_err, mean_dist_err_list, index[i], color_map.get(algo, "black"))

            mean_sub = np.nanmean(np.take(mean_subgoals_finished_list, index[i], 0), 0)
            ax5.set_ylim([0, 100])
            ax_plot(ax5, steps, mean_sub, color_map.get(algo, "black"), algo)

            # stacked bars
            ax = fig2.add_subplot(ceiling(plot_num / 2), ceiling(plot_num / 2), counter + 1)
            ax.set_ylim(0, 100)

            # choose first non-None subgoal matrix in this group
            sub_mats = [mean_subgoals_steps_list[j] for j in index[i] if mean_subgoals_steps_list[j] is not None]
            acts = None
            if index[i]:
                acts = acts_list[index[i][0]] if index[i][0] < len(acts_list) else None

            if sub_mats and acts:
                mean_sub_steps = np.nanmean(np.stack(sub_mats, axis=0), axis=0)  # (K, T)
                K = min(mean_sub_steps.shape[0], len(acts))
                mean_sub_steps = mean_sub_steps[:K, :]
                bottom = np.zeros(len(steps), dtype=np.float32)

                # width heuristic
                width = (steps[1] - steps[0]) if len(steps) >= 2 else float(max_steps_cfg or 512)

                for k in range(K):
                    p = ax.bar(x=steps, height=mean_sub_steps[k],
                               color=color_map_acts.get(acts[k], "black"),
                               label=f"{acts[k]}",
                               bottom=bottom,
                               width=-width,
                               align='edge',
                               edgecolor='black')
                    bottom = bottom + mean_sub_steps[k]

                # unused
                unused = np.maximum(0.0, 100.0 - bottom)
                ax.bar(x=steps, height=unused, color="white", label="unused steps",
                       bottom=bottom, width=-width, align='edge', edgecolor='black')
            else:
                ax.text(0.5, 0.5, "No 'mean subgoal steps' in evaluation logs",
                        ha="center", va="center", transform=ax.transAxes)

            task = configs[index[i][0]].get("task_type", "") if index[i] else ""
            ax_set(ax, f"Subgoal steps over episode for algo: {algo}, task: {task}", "Mean steps (%)")
            counter += 1

    else:
        # single plot
        steps = steps_list[0]
        cfg = configs[0]
        algo = cfg.get("algo", "algo")

        ax_plot(ax1, steps, success_list[0], color_map.get(algo, "black"), algo)
        ax_plot(ax2, steps, mean_reward_list[0], color_map.get(algo, "black"), algo)
        ax2.plot(steps, std_reward_list[0], alpha=0.8, color=color_map.get(algo, "black"),
                 linestyle='dotted', linewidth=1.5, marker='o',
                 markerfacecolor=color_map.get(algo, "black"), markersize=2, label=f"{algo} std")
        ax_plot(ax3, steps, mean_steps_list[0], color_map.get(algo, "black"), algo)
        ax_plot(ax4, steps, mean_dist_err_list[0], color_map.get(algo, "black"), algo)
        ax5.set_ylim([0, 100])
        ax_plot(ax5, steps, mean_subgoals_finished_list[0], color_map.get(algo, "black"), algo)

        ax = fig2.add_subplot(1, 1, 1)
        ax.set_ylim(0, 100)

        sub = mean_subgoals_steps_list[0]
        acts = acts_list[0] if acts_list else []

        if sub is not None and acts:
            K = min(sub.shape[0], len(acts))
            sub = sub[:K, :]
            bottom = np.zeros(len(steps), dtype=np.float32)
            width = (steps[1] - steps[0]) if len(steps) >= 2 else float(max_steps_cfg or 512)

            for k in range(K):
                ax.bar(x=steps, height=sub[k],
                       color=color_map_acts.get(acts[k], "black"),
                       label=f"{acts[k]}",
                       bottom=bottom,
                       width=-width,
                       align='edge',
                       edgecolor='black')
                bottom = bottom + sub[k]
            unused = np.maximum(0.0, 100.0 - bottom)
            ax.bar(x=steps, height=unused, color="white", label="unused steps",
                   bottom=bottom, width=-width, align='edge', edgecolor='black')
        else:
            ax.text(0.5, 0.5, "No 'mean subgoal steps' in evaluation logs",
                    ha="center", va="center", transform=ax.transAxes)

        ax_set(ax, f"Subgoal steps over episode for algo: {algo}, task: {cfg.get('task_type', '')}", "Mean steps (%)")

    # Set titles for axes
    ax_set(ax1, 'Success rate', 'Successful episodes (%)')
    ax_set(ax2, 'Mean/std rewards', 'Mean/std rewards')
    ax_set(ax3, 'Mean steps', 'Steps')
    ax_set(ax4, 'Mean distance error', 'Error')
    ax_set(ax5, 'Finished subgoals in %', 'Subgoals')

    fig1.tight_layout()
    fig2.tight_layout()

    # GT vs Decider
    try:
        if last_cfg is not None:
            task = last_cfg.get("task_type", "")
            acts = [dict_acts[c] for c in str(task) if c in dict_acts]
            plot_gt_vs_decider_bar(experiment_dirs[0], acts, out_dir, last_cfg)
    except Exception as e:
        print(f"Failed to create GT vs Decider plot: {e}")

    # Save figures
    logdir_name = cut_before_last_slash(last_cfg.get("logdir", "experiment")) if last_cfg else "experiment"
    fig1_path = os.path.join(out_dir, f"{logdir_name}_train.png")
    fig2_path = os.path.join(out_dir, f"{logdir_name}_goals.png")
    fig1.savefig(fig1_path, dpi=150)
    fig2.savefig(fig2_path, dpi=150)
    print(f"Figures saved to:\n  {fig1_path}\n  {fig2_path}")
    plt.show()


if __name__ == "__main__":
    main()
