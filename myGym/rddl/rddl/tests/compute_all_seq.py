from collections import deque
import sys
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from compute_max_seq import get_sequence_from_csv, action_set
from sequence_metrics import levenshtein_ratio, hamming, jaro_winkler, longest_common_subsequence, longest_common_substring

starting_action_set = {
    "Approach"
}

SEQUENCE_LENGTH = 5
RES_QUE = deque()


def generate_all_sequences(actions, starting_actions, sequence_length):
    def explore_all(current_action, depth, sequence):
        if depth == sequence_length:
            RES_QUE.append(sequence + [current_action])
            return 1
        else:
            return sum([explore_all(action, depth + 1, sequence + [current_action]) for action in actions])

    total_sequences = 0
    for start_action in starting_actions:
        number_of_sequences = explore_all(start_action, 1, [])
        print(f"Found {number_of_sequences} sequences for {start_action} as starting action.")
        total_sequences += number_of_sequences
    print(f"Total number of sequences: {total_sequences}")
    return RES_QUE


if __name__ == "__main__":
    all_sequences = generate_all_sequences(action_set, starting_action_set, SEQUENCE_LENGTH)
    all_seq_map = {tuple(k): v for v, k in enumerate(all_sequences)}
    inv_map = {v: list(k) for k, v in all_seq_map.items()}
    all_sequence_df = pd.DataFrame(inv_map.values(), index=inv_map.keys())
    all_sequence_df.to_csv(f"all-sequences-len_{SEQUENCE_LENGTH}.csv")

    print("Computing pair-wise distances...")
    distances_per_metric = {}
    for metric in tqdm([levenshtein_ratio, hamming, jaro_winkler, longest_common_subsequence, longest_common_substring], desc="Metrics"):
        pair_wise_distances = {}
        for i, sequence_i in enumerate(all_sequences):
            for j, sequence_j in enumerate(all_sequences):
                if i < j:
                    pair_wise_distances[(i, j)] = metric(sequence_i, sequence_j)

        distances_per_metric[metric.__name__] = pair_wise_distances

    distance_matrix_df = pd.DataFrame(distances_per_metric)
    distance_matrix_df.to_csv(f"distances-len_{SEQUENCE_LENGTH}.csv")
