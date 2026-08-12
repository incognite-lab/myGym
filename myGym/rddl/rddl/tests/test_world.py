import datetime
import sys
import numpy as np
from tqdm import tqdm
import yaml
import testing_utils
from rddl.predicates import Near, OnTop
from rddl.rule_book import ExclusivityRule
import traceback as tb
from rddl import Entity, Variable
from rddl.entities import Gripper, Location, ObjectEntity
from rddl.rddl_sampler import RDDLWorld, Weighter
import pandas as pd
from memory_profiler import profile
import tracemalloc as tm
# from scipy.stats import wasserstein_distance
from sequence_metrics import (levenshtein_ratio, longest_common_subsequence, longest_common_substring, hamming, jaro_winkler,
                              average_over_repeats, average_over_repeats_pooled, compute_avg_distance)
import multiprocess as mp
import argparse


def test_world(sequence_lengths: int | list[int] = 4, n_repeats: int = 1, output_path: str | None = None):
    action_record = {}
    if isinstance(sequence_lengths, int):
        sequence_lengths = [sequence_lengths]

    for sequence_length in sequence_lengths:
        print(f"Sequence length: {sequence_length}")
        record_sequence = []
        for r in range(n_repeats):
            print(f"Repeat {r}")

            rddl_world = RDDLWorld(

            )
            task = rddl_world.sample_world(sequence_length=sequence_length)
            actions = task.get_actions()
            variables = task.gather_objects()

            record_rep = []
            for a in actions:
                record_rep.append({
                    "action": str(a),
                    "action_objects": {k: v.name for k, v in a.variables.items()}
                })

            record_sequence.append({
                "action_list": [a.__class__.__name__ for a in task.get_actions()],
                "sequence": record_rep,
                "all_objects": {v.name: v.type.__name__ for v in variables}
            })

        action_record[sequence_length] = record_sequence

    if output_path is not None:
        with open(output_path, "w") as f:
            yaml.dump(action_record, f)


        # str_actions = '\n\t'.join([repr(a) for a in actions])
        # print(f"Actions:\n\t{str_actions}")
        # str_variables = '\n\t'.join([repr(v) for v in variables])
        # print(f"Variables:\n\t{str_variables}")


def format_averaged_results(mean, std, mean_precision=2, std_precision=4):
    return f"{mean:.{mean_precision}f} ± {std:.{std_precision}f}"


def dict_average_results(name, mean, std, mean_precision=2, std_precision=4):
    return {
        name: f"{mean:.{mean_precision}f}",
        f"{name[:3]} std": f"{std:.{std_precision}f}"
    }


def test_world_generator():
    rddl_world = RDDLWorld()
    # rddl_world.set_allowed_entities(
    #     [Gripper, Apple, Tuna]
    # )
    # rddl_world.set_allowed_actions([Approach, Withdraw, Grasp, Drop, Move])
    # rddl_world.set_allowed_predicates([IsReachable])

    n_samples = 30
    # rddl_task = rddl_world.sample_world(n_samples)

    # for o in rddl_task.objects:
    #     my_o = o.instantiate()
    #     o.bind(my_o)

    # reward: float = rddl_task.current_reward()


    actions = []


    gen = rddl_world.sample_generator(n_samples)

    while True:
        try:
            action = next(gen)
        except StopIteration:
            break
        print(f"Generated action: {action}")
        print("World state after action:")
        rddl_world.show_world_state()
        print("")
        actions.append(action)

    variables = rddl_world.get_created_variables()

    str_actions = '\n\t'.join([repr(a) for a in actions])
    print(f"Actions:\n\t{str_actions}")
    str_variables = '\n\t'.join([repr(v) for v in variables])
    print(f"Variables:\n\t{str_variables}")
    print("> Initial state:")
    rddl_world.show_initial_world_state()
    print("> Goal state:")
    rddl_world.show_goal_world_state()


def test_world_generator_random_reject():
    rddl_world = RDDLWorld()

    n_samples = 100
    actions = []
    gen = rddl_world.sample_generator(n_samples)

    for _ in range(n_samples):
        action = next(gen)
        if np.random.rand() < 0.5:
            action = gen.send(False)
            print(f"Rejected action: {action}, regenerating...")
        else:
            print(f"Generated action: {action}")
            actions.append(action)

    variables = rddl_world.get_created_variables()

    str_actions = '\n\t'.join([repr(a) for a in actions])
    print(f"Actions:\n\t{str_actions}")
    str_variables = '\n\t'.join([repr(v) for v in variables])
    print(f"Variables:\n\t{str_variables}")


def yielder():
    print("function start")
    for i in range(10):
        print("pre-yield")
        data = yield i
        print("post-yield", data)
    print("function end")


@profile
def test_sampling_eff_multi():
    # experiment settings >>>
    number_of_repeats = 1  # number of times to repeat each experiment
    save_sequences = True  # whether to save unique sequences to a csv (for later analysis)

    # generate this many sequences, multiple values can be inserted
    # list_of_number_of_sequences = [50, 100, 500, 1000, 2000, 5000]
    # list_of_number_of_sequences = [2000]
    list_of_number_of_sequences = [100]

    # generate sequences of this length, multiple values can be inserted
    # list_of_sequence_lengths = [5, 10, 15, 20]
    # list_of_sequence_lengths = [20]
    list_of_sequence_lengths = [4]

    # test the following sampling modes
    modes = [Weighter.MODE_WEIGHT | Weighter.MODE_SEQUENCE | Weighter.MODE_MAX_NOISE]
    # modes = [Weighter.MODE_RANDOM]
    # modes += [Weighter.MODE_WEIGHT, Weighter.MODE_SEQUENCE]
    # modes += [Weighter.MODE_WEIGHT | Weighter.MODE_MAX_NOISE]
    # modes += [Weighter.MODE_SEQUENCE | Weighter.MODE_MAX_NOISE]
    # modes += [Weighter.MODE_WEIGHT | Weighter.MODE_SEQUENCE | Weighter.MODE_MAX_NOISE]
    # modes += [Weighter.MODE_SEQUENCE | Weighter.MODE_RANDOM]
    # modes += [Weighter.MODE_WEIGHT | Weighter.MODE_RANDOM]
    # modes += [Weighter.MODE_WEIGHT | Weighter.MODE_SEQUENCE | Weighter.MODE_RANDOM]

    # <<< end of settings
    random_seeds = np.random.randint(0, 2**32 - 1, number_of_repeats, dtype=np.uint32)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_filename = f"geneff_{number_of_repeats}-reps-{timestamp}.csv"
    result_table = None
    mode_names = {
        Weighter.MODE_INITIAL: "Initial",
        Weighter.MODE_WEIGHT: "Weight",
        Weighter.MODE_SEQUENCE: "Sequence",
        Weighter.MODE_RANDOM: "Random",
        Weighter.MODE_MAX_NOISE: "Max_Noise",
    }

    tm.start()
    # [Approach, Withdraw, Grasp, Drop, Move, Rotate, Follow, Transform]
    rddl_world = RDDLWorld(
        # action_weights=[0.1, 0.1, 10, 10, 10, 10, 10, 10],
        # action_weights=[0.1, 0.1, 0.1, 0.1, 100, 0.1, 0.1, 0.1],
    )

    sampling_attempts_pbar = tqdm(list_of_number_of_sequences, position=0, desc="Number of sequences")
    for n_sampling_attempts in sampling_attempts_pbar:
        sampling_attempts_pbar.set_description(f"Number of sequences [{n_sampling_attempts}]")
        tqdm.write("=============================")
        tqdm.write(f">>> NUMBER OF SEQUENCES: {n_sampling_attempts}")
        tqdm.write("=============================")
        n_sampling_attempts_pbar = tqdm(list_of_sequence_lengths, position=1, desc="Sequence length", leave=False)
        for n_samples_per_attempt in n_sampling_attempts_pbar:
            n_sampling_attempts_pbar.set_description(f"Sequence length [{n_samples_per_attempt}]")
            tqdm.write("-----------------------------")
            tqdm.write(f">>> SEQUENCE LENGTH: {n_samples_per_attempt}")
            tqdm.write("-----------------------------")

            try:
                mode_pbar = tqdm(modes, position=2, desc="Mode", leave=False)
                for mode in mode_pbar:
                    tqdm.write("\n>>>>>>>>>>>>>>>>>>>>")
                    tqdm.write("Evaluating:")
                    tqdm.write(f"Mode: {mode} [" + ' | '.join([mn for mv, mn in mode_names.items() if mv & mode]) + "]")
                    mode_pbar.set_description(f"Mode [{mode}]")
                    found_uq_sequences_per_repeat = []
                    for repeat_seed in tqdm(random_seeds, position=3, desc="Repeat", leave=False):
                        RDDLWorld.set_seed(repeat_seed)
                        rddl_world.reset_weights(mode)
                        found_uq_sequences = {}
                        mode = rddl_world.weighter.mode  # reassign to make sure the correct mode is set

                        for _ in tqdm(range(n_sampling_attempts), position=4, desc="Sequence", leave=False):
                            gen = rddl_world.sample_generator(n_samples_per_attempt)
                            actions = []
                            for _ in tqdm(range(n_samples_per_attempt), position=5, desc="Sample", leave=False):
                                try:
                                    action = next(gen)
                                except StopIteration:
                                    break
                                actions.append(action)

                            h = tuple(a.__class__.__name__ for a in actions)
                            if h in found_uq_sequences:
                                found_uq_sequences[h] += 1
                            else:
                                found_uq_sequences[h] = 1

                        if save_sequences:
                            # convert found_uq_sequences to pandas DataFrame and save it as csv
                            df = pd.DataFrame.from_dict({"n": list(found_uq_sequences.values())} | {f"action_{ai}": a for ai, a in enumerate(zip(*[s for s in found_uq_sequences.keys()]))})
                            df.to_csv(f"gseq-{timestamp}_len_{n_samples_per_attempt}_mode_{mode}_n_{n_sampling_attempts}_rs-{repeat_seed}.csv")

                        found_uq_sequences_per_repeat.append(found_uq_sequences)

                    tqdm.write("Computing metrics...")
                    tqdm.write("---------------------")
                    r_unique_seq_list = list(list(fseq.keys()) for fseq in found_uq_sequences_per_repeat)
                    r_unique_seq_values = list(list(fseq.values()) for fseq in found_uq_sequences_per_repeat)
                    r_unique_seq_lens = list(len(fseq) for fseq in found_uq_sequences_per_repeat)
                    avg_len_found_uq_sequences = np.mean(r_unique_seq_lens)
                    efficiency = average_over_repeats(r_unique_seq_lens, lambda x, n=n_sampling_attempts: x / n)
                    avg_repeats = average_over_repeats(r_unique_seq_values, np.mean)

                    tqdm.write(f"Number of unique sequences: {avg_len_found_uq_sequences:0.02f} / {n_sampling_attempts} (total attempts, {n_samples_per_attempt} samples per attempt)")
                    tqdm.write(f"Efficiency: {format_averaged_results(*efficiency)}")
                    tqdm.write(f"Average repeat: {format_averaged_results(*avg_repeats)}")
                    tqdm.write("<<<<<<<<<<<<<<<<<<<<<\n")

                    rec_dict = {
                        "mode": mode,
                        "mode_name": ' | '.join([mn for mv, mn in mode_names.items() if mv & mode]),
                        "n_sequences": n_sampling_attempts,
                        "sequence_length": n_samples_per_attempt,
                        "uq_sequences (+)": f"{avg_len_found_uq_sequences:0.01f}",
                        "uq std": f"{np.std(r_unique_seq_lens):0.02f}",
                    }
                    rec_dict |= dict_average_results("efficiency (+)", *efficiency, mean_precision=3, std_precision=3)
                    rec_dict |= dict_average_results("repeats_avg (-)", *avg_repeats, mean_precision=1, std_precision=2)
                    if n_sampling_attempts < 2001:
                        rec_dict |= dict_average_results("levenshtein (-)", *average_over_repeats(r_unique_seq_list, compute_avg_distance, distance_func=levenshtein_ratio), mean_precision=3, std_precision=4)
                        rec_dict |= dict_average_results("jaro_winkler (-)", *average_over_repeats(r_unique_seq_list, compute_avg_distance, distance_func=jaro_winkler, prefix_weight=0.24), mean_precision=3, std_precision=4)
                        rec_dict |= dict_average_results("hamming_ratio (+)", *average_over_repeats(r_unique_seq_list, compute_avg_distance, distance_func=lambda x, y: hamming(x, y) / n_samples_per_attempt), mean_precision=3, std_precision=4)
                    if n_sampling_attempts < 1001:
                        rec_dict |= dict_average_results("lcs_ratio (-)", *average_over_repeats_pooled(POOL, r_unique_seq_list, compute_avg_distance, distance_func=lambda x, y: longest_common_subsequence(x, y) / n_samples_per_attempt), mean_precision=3, std_precision=4)
                        rec_dict |= dict_average_results("lc_substr_ratio (-)", *average_over_repeats_pooled(POOL, r_unique_seq_list, compute_avg_distance, distance_func=lambda x, y: longest_common_substring(x, y) / n_samples_per_attempt), mean_precision=3, std_precision=4)
                    new_rec = pd.DataFrame(rec_dict, index=[0])

                    # from timeit import repeat
                    # # timeit(lambda: average_over_repeats_pooled(POOL, r_unique_seq_list, compute_avg_distance, distance_func=levenshtein_ratio), number=100)
                    # print("average levenshtein runtime (10 reps):\n",
                    #       str(np.mean(repeat(lambda: average_over_repeats(r_unique_seq_list, compute_avg_distance, distance_func=levenshtein_ratio), number=10, repeat=3))))
                    #     #   str(np.mean(repeat(lambda: average_over_repeats(r_unique_seq_list, compute_avg_distance, distance_func=lambda x, y: longest_common_subsequence(x, y) / n_samples_per_attempt), number=100, repeat=3))))
                    #     #   str(np.mean(repeat(lambda: average_over_repeats(r_unique_seq_list, compute_avg_distance, distance_func=lambda x, y: longest_common_substring(x, y) / n_samples_per_attempt), number=100, repeat=3))))

                    # print("average pooled levenshtein runtime (10 reps):\n",
                    #       str(np.mean(repeat(lambda: average_over_repeats_pooled(POOL, r_unique_seq_list, compute_avg_distance, distance_func=levenshtein_ratio), number=10, repeat=3))))
                    # print("average pooled lcs runtime (10 reps):\n",
                    #       str(np.mean(repeat(lambda: average_over_repeats_pooled(POOL, r_unique_seq_list, compute_avg_distance, distance_func=lambda x, y: longest_common_subsequence(x, y) / n_samples_per_attempt), number=10, repeat=3))))
                    # print("average pooled lcsubstr runtime (1 rep):\n",
                    #     #   str(np.mean(repeat(lambda: average_over_repeats_pooled(POOL, r_unique_seq_list, compute_avg_distance, distance_func=levenshtein_ratio), number=100, repeat=3))))
                    #     #   str(np.mean(repeat(lambda: average_over_repeats_pooled(POOL, r_unique_seq_list, compute_avg_distance, distance_func=lambda x, y: longest_common_subsequence(x, y) / n_samples_per_attempt), number=100, repeat=3))))
                    #       str(np.mean(repeat(lambda: average_over_repeats_pooled(POOL, r_unique_seq_list, compute_avg_distance, distance_func=lambda x, y: longest_common_substring(x, y) / n_samples_per_attempt), number=1, repeat=3))))

                    if result_table is None:
                        result_table = new_rec
                    else:
                        result_table = pd.concat([result_table, new_rec], ignore_index=True)
                    result_table.to_csv(out_filename)
            except BaseException as e:
                print(f"Error occured during # sequences {n_sampling_attempts} & # samples {n_samples_per_attempt}:\n{e}")
                tb.print_exc()
                raise e
                continue

            current_mem_usage, peak_mem_usage = tm.get_traced_memory()
            tqdm.write(f">>> Memory usage: current = {current_mem_usage/1e6:0.2f} MB | peak = {peak_mem_usage/1e6:0.2f} MB")
            tm.reset_peak()

            tqdm.write(f"<<< ------- # of samples {n_sampling_attempts}")
        tqdm.write(f"<<< ======= # of sequences {n_sampling_attempts}")

    result_table = pd.concat([
        result_table,
        pd.DataFrame({
            "mode_name": "",
        }, index=[0]),
        pd.DataFrame({
            "mode_name": f"each row repeated {number_of_repeats} times",
        }, index=[0]),
        pd.DataFrame({
            "mode_name": "(+) means higher values are better",
        }, index=[0]),
        pd.DataFrame({
            "mode_name": "(-) means lower values are better",
        }, index=[0])
    ], ignore_index=True)
    result_table.to_csv(out_filename)

    mem_snap = tm.take_snapshot()
    for stat in mem_snap.statistics('lineno'):
        tbfmt = stat.traceback.format()
        if 'debug' not in tbfmt[0] and (stat.size > 1e5 or stat.count > 1000):
            print(tbfmt)
            print(stat)


def test_recursive_generator():
    rddl_world = RDDLWorld(
        sample_single_object_per_class=True
    )
    rddl_world.rule_book.add_rule(
        ExclusivityRule(OnTop, Near)
    )
    n_samples = 5
    sequence_length = 10
    n_regen_objects = 0

    gen = rddl_world.recurse_generator(sequence_length)

    def pprint_objects(task):
        print("\n".join([f"\t - {o.type.__name__}" for o in task.gather_objects()]))

    for _ in range(n_samples):
        task = next(gen)
        print("Action sequence:")
        print("\n".join([f"\t{a}" for a in task.get_actions()]))
        print("Initial state:")
        task.initial_state.show_table()
        print("Final state:")
        task.final_state.show_table()
        print("Object set 0:")
        pprint_objects(task)
        for os_idx in range(n_regen_objects):
            task.regenerate_objects()
            print(f"Object set {os_idx + 1}:")
            pprint_objects(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-style", "-o", action="store_true", help="Use old-style sampling - randomized. New style is recursive and deterministic.")
    args = parser.parse_args()
    # test_world()
    # test_world_generator()
    # test_sampling_eff()

    # test_world()
    test_world(sequence_lengths=list(range(2, 11)), n_repeats=10, output_path="test_world.yaml")
    sys.exit(0)

    if args.old_style:
        global POOL
        POOL = mp.Pool(processes=mp.cpu_count())
        test_sampling_eff_multi()
        POOL.terminate()
    else:
        test_recursive_generator()
