from enum import unique
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os

METRICS = {
    "efficiency": "efficiency",
    "eff": "efficiency",
    "levenshtein": "levenshtein",
    "lev": "levenshtein",
    "jaro_winkler": "jaro_winkler",
    "jaro": "jaro_winkler",
    "hamming": "hamming_ratio",
    "ham": "hamming_ratio",
    "longest_common_subsequence": "lcs_ratio",
    "lcs": "lcs_ratio",
    "substring": "lc_substr_ratio",
    "sub": "lc_substr_ratio",
    "all": "all",
}

N_SEQUENCES = "n_sequences"
SEQUENCE_LENGTH = "sequence_length"
MODE_NAME = "mode_name"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("metric", type=str, choices=METRICS.keys())
    parser.add_argument("--list", "-l", action="store_true", help="List all metrics.")
    mex_group = parser.add_mutually_exclusive_group()
    mex_group.add_argument("--group-by-sequence-length", "-s", action="store_true")
    mex_group.add_argument("--group-by-n-sequences", "-n", action="store_true")
    parser.add_argument("--output", "-o", type=str, default=None)

    args = parser.parse_args()

    if args.list:
        for metric_name in METRICS.keys():
            print(metric_name)
        exit()

    if not os.path.exists(args.input):
        raise ValueError(f"File {args.input} does not exist or the path to it is incorrect!")

    df = pd.read_csv(args.input)
    if N_SEQUENCES not in df.columns:
        raise ValueError("'n_sequences' is not in columns. Maybe this is not a test world results file or incompatible version?")
    if SEQUENCE_LENGTH not in df.columns:
        raise ValueError("'sequence_length' is not in columns. Maybe this is not a test world results file or incompatible version?")

    column_names = df.columns.values

    if args.group_by_sequence_length:
        df = df.groupby([MODE_NAME, SEQUENCE_LENGTH])
        aggregation_name = "n_sequences"
        x_label = "sequence_length"
    elif args.group_by_n_sequences:
        df = df.groupby([MODE_NAME, N_SEQUENCES])
        aggregation_name = "sequence_length"
        x_label = "n_sequences"
    else:
        raise ValueError("Please select a group by option.")
    aggregated_df = df.agg(["mean", "std"])

    if args.output is not None:
        output_name = f"{args.output}_{os.path.splitext(os.path.basename(args.input))[0]}_{{metric}}.png"

    if args.metric == "all":
        list_of_metrics = set(METRICS.values()) - {'all'}
    else:
        list_of_metrics = [METRICS[args.metric]]

    # generate unique, easily distinguishable colors for each metric in list_of_metrics
    unique_modes = aggregated_df.index.get_level_values(0).unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_modes)))

    for short_metric_name in list_of_metrics:
        for i, m in enumerate(column_names):
            if short_metric_name in m:
                metric_name = m
                metric_index = i
                metric_std_name = column_names[i + 1]
                break
        else:
            raise ValueError(f"Could not find metric '{short_metric_name}' in columns: {column_names}")

        aggregated_metric, aggregated_std = aggregated_df[metric_name], aggregated_df[metric_std_name]["mean"]

        fig = plt.figure()
        ax = plt.gca()
        # plot one line plot for each "mode_name", with error bars
        for mode_name, plot_color in zip(unique_modes, colors):
            mode_values = aggregated_metric.loc[mode_name]
            x = mode_values.index.values
            y = mode_values["mean"].values
            y_std = mode_values["std"].values
            agg_original_std = aggregated_std.loc[mode_name].values
            ax.plot(x, y, label=mode_name, color=plot_color)
            ax.fill_between(x, y - y_std, y + y_std, alpha=0.05, interpolate=True, color=plot_color)
            ax.fill_between(x, y - agg_original_std, y + agg_original_std, alpha=0.2, interpolate=True, color=plot_color)

        ax.set_xticks(x)
        plt.xlabel(x_label)
        plt.ylabel(short_metric_name)
        plt.title(f"{metric_name} aggregated over {aggregation_name}")
        plt.legend(loc="upper left")
        fig.set_size_inches(16, 9)
        fig.set_dpi(120)
        fig.tight_layout(pad=0.5)

        if args.output is not None:
            plt.savefig(output_name.format(metric=short_metric_name))
        else:
            plt.show()

        plt.close()
