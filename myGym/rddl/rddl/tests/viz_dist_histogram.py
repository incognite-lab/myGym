import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from compute_max_seq import get_sequence_from_csv


SEQUENCE_LENGTH = 5
compare_to = "gseq-20240625-113130_len_5_mode_14_n_2000_rs-650722818.csv"

if __name__ == "__main__":
    distance_df = pd.read_csv(f"./tests/distances-len_{SEQUENCE_LENGTH}.csv", index_col=[0, 1])
    all_sequences = pd.read_csv(f"./tests/all-sequences-len_{SEQUENCE_LENGTH}.csv", index_col=0)
    viable_sequences = get_sequence_from_csv(f"./sequences-len_{SEQUENCE_LENGTH}.csv")
    if compare_to is not None:
        comparison_sequences = pd.read_csv(f"tests/{compare_to}", index_col=0).iloc[:, 1:].to_numpy()

    all_seq_map = {tuple(k): v for v, *k in all_sequences.to_records()}
    viable_seq_map = {tuple(k): all_seq_map[tuple(k)] for k in viable_sequences}
    compare_seq_map = {tuple(k): all_seq_map[tuple(k)] for k in comparison_sequences}

    metrics = list(distance_df.columns)

    for metric in tqdm(metrics, desc="Metrics"):
        distances = distance_df[metric]
        viable_distances = distances[viable_seq_map.values()]
        comp_distances = distances[compare_seq_map.values()]
        fig = plt.figure()
        ax = plt.gca()
        ax.hist([distances, viable_distances, comp_distances], density=True,
                bins=10, alpha=0.7, color=["black", "red", "blue"], histtype="bar",
                label=["all", "viable", "sampled"])
        ax.legend()
        ax.set_title(metric)
        fig.tight_layout()
        fig.savefig(f"tests/dist-hist-{metric}-len_{SEQUENCE_LENGTH}.png")
