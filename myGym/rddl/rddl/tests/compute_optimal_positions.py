import pandas as pd
import numpy as np
import networkx as nx
# from ogdf_python import *
# from forceatlas2py import ForceAtlas2
from fa2 import ForceAtlas2

from tqdm import tqdm

sequence_length = 5
# cppinclude("ogdf/energybased/FMMMLayout.h")


def optimize_positions_nx(nodes, pair_wise_distances):
    # Create a graph from the pair-wise distances
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for (i, j), distance in pair_wise_distances.items():
        G.add_edge(i, j, weight=distance)

    # Perform the force-directed layout
    pos = nx.spring_layout(G, weight='weight', iterations=2000)

    return pos


def optimize_positions_atlas(nodes, pair_wise_distances):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for (i, j), distance in pair_wise_distances.items():
        G.add_edge(i, j, weight=distance)

    forceatlas2 = ForceAtlas2(
        # Behavior alternatives
        outboundAttractionDistribution=True,  # Dissuade hubs
        linLogMode=False,  # NOT IMPLEMENTED
        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
        edgeWeightInfluence=1.0,

        # Performance
        jitterTolerance=1.0,  # Tolerance
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        multiThreaded=False,  # NOT IMPLEMENTED

        # Tuning
        scalingRatio=1.0,
        strongGravityMode=False,
        gravity=1.0,

        # Log
        verbose=True
    )

    positions = forceatlas2.forceatlas2_networkx_layout(
        G=G,
        pos=None,
        iterations=2000
    )

    return positions

# def optimize_positions_ogdf(nodes, pair_wise_distances):
#     ogdf.setSeed(0)
#     # Create a graph from the pair-wise distances
#     G = ogdf.Graph()
#     node_dict = {node: G.newNode(node) for node in nodes}
#     GA = ogdf.GraphAttributes(G, ogdf.GraphAttributes.all)
#     for (i, j), distance in pair_wise_distances.items():
#         e = G.newEdge(node_dict[i], node_dict[j])
#         GA.doubleWeight[e] = distance
#     # Perform the force-directed layout
#     pos = G.force_directed_layout()

#     return pos


if __name__ == "__main__":
    distance_df = pd.read_csv(f"distances-len_{sequence_length}.csv", index_col=[0, 1])
    metrics = list(distance_df.columns)

    node_range = list(range(min(distance_df.index.min()), max(distance_df.index.max()) + 1))

    # Optimize positions
    print("Optimizing positions...")
    positions_per_metric = {}
    optimal_node_positions = {}
    for metric in tqdm(metrics, desc="Metrics"):
        pair_wise_distances = distance_df[metric].to_dict()
        # optimal_node_positions[metric] = optimize_positions_nx(node_range, pair_wise_distances)
        optimal_node_positions[metric] = optimize_positions_atlas(node_range, pair_wise_distances)
        # optimal_node_positions[metric] = optimize_positions_ogdf(node_range, pair_wise_distances)
        positions_per_metric[metric] = pair_wise_distances

    optimal_node_positions_df = pd.DataFrame(optimal_node_positions)
    optimal_node_positions_df.to_csv(f"positions-len_{sequence_length}.csv")
