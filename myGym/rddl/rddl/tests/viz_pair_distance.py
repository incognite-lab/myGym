from collections import deque
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from compute_max_seq import get_sequence_from_csv, action_set, starting_action_set
from PIL import Image
from sequence_metrics import levenshtein_ratio


SEQUENCE_LENGTH = 5
IMAGE_SIDE = 100
RES_QUE = deque()
BASE_COLOR = 180
compare_to = "gseq-20240625-113130_len_5_mode_14_n_2000_rs-650722818.csv"


def visualize_positions_as_image(pos, width, height, highlight_red=[], highlight_green=[]):
    # image = Image.new('RGB', (width, height), (0, 0, 0))
    # pixels = image.load()
    np_image = np.zeros((width, height, 3), dtype=np.uint16)

    # Set the pixel color based on the position
    for node, position in pos.items():
        x = int(position[0] * (width - 1))
        y = int(position[1] * (height - 1))
        if node in highlight_green:
            np_image[x, y, 1] += 1
        elif node in highlight_red:
            np_image[x, y, 0] += 1
        else:
            np_image[x, y, 2] += 1

    max_value = np_image.max()
    where_value = np_image > 0
    where_empty = np.all(np.logical_not(where_value), axis=2)
    np_image[where_value] = BASE_COLOR + ((255 - BASE_COLOR) / max_value) * np_image[where_value]
    np_image[where_empty] = 255

    image = Image.fromarray(np.clip(np_image, 0, 255).astype(np.uint8))

    return image


if __name__ == "__main__":
    viable_sequences = get_sequence_from_csv(f"tests/sequences-len_{SEQUENCE_LENGTH}.csv")
    all_sequences = pd.read_csv(f"tests/all-sequences-len_{SEQUENCE_LENGTH}.csv", index_col=0)
    all_seq_map = {tuple(k): v for v, *k in all_sequences.to_records()}
    viable_seq_map = {tuple(k): all_seq_map[tuple(k)] for k in viable_sequences}

    if compare_to is not None:
        comparison_sequences = pd.read_csv(f"tests/{compare_to}", index_col=0).iloc[:, 1:].to_numpy()
        compare_seq_map = {tuple(k): all_seq_map[tuple(k)] for k in comparison_sequences}
        sampled_positions = list(compare_seq_map.values())
    else:
        sampled_positions = []

    positions_per_metric = pd.read_csv(f"tests/positions-len_{SEQUENCE_LENGTH}.csv", index_col=0).to_dict()

    viable_positions = list(viable_seq_map.values())
    print("Visualizing positions...")
    tupleware = next(iter(positions_per_metric.values()))[0].startswith('(')
    for metric, raw_positions in tqdm(positions_per_metric.items(), desc="Metrics"):
        if tupleware:
            positions = {k: np.fromstring(s.strip('()'), sep=', ') for k, s in raw_positions.items()}
        else:
            positions = {k: np.fromstring(s.strip('[]'), sep=' ') for k, s in raw_positions.items()}
        # rescale positions
        pos_array = np.asarray(list(positions.values()))
        min_pos, max_pos = pos_array.min(axis=0), pos_array.max(axis=0)
        positions = {k: (v - min_pos) / (max_pos - min_pos) for k, v in positions.items()}
        image = visualize_positions_as_image(positions, IMAGE_SIDE, IMAGE_SIDE, viable_positions, sampled_positions)
        image.save(f"tests/graph-{metric}-len_{SEQUENCE_LENGTH}.png")
