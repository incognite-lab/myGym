from typing import Callable, Iterable, Sized
from Levenshtein import distance as levenshtein_distance
from Levenshtein import ratio as levenshtein_ratio
from Levenshtein import hamming
from Levenshtein import jaro_winkler
import multiprocess as mp
from functools import partial
import numpy as np

"""
Metrics:
levenshtein_ratio: Levenshtein ratio [0, 1]
longest_common_subsequence: Longest common subsequence [0, N] where N is the length of the longer sequence
longest_common_substring: Longest common substring [0, N] where N is the length of the longer sequence
hamming: Hamming distance [0, N] where N is the length of the sequence (only works for sequences of the same length)
jaro_winkler: Jaro-Winkler distance [0, 1]
"""


def longest_common_substring(X: Sized, Y: Sized) -> int:
    """Computes longest common substring between X and Y: https://en.wikipedia.org/wiki/Longest_common_substring

    Args:
        X (list): A string or list of comparable entities.
        Y (list): A string or list of comparable entities.

    Returns:
        int: The length of the longest common substring between X and Y.
    """
    def lc_sub_str(s, t, n, m):

        # Create DP table
        dp = [[0 for _ in range(m + 1)] for _ in range(2)]
        res = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if (s[i - 1] == t[j - 1]):
                    dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
                    if (dp[i % 2][j] > res):
                        res = dp[i % 2][j]
                else:
                    dp[i % 2][j] = 0
        return res

    n, m = len(X), len(Y)
    return lc_sub_str(X, Y, n, m)


def longest_common_subsequence(X: Sized, Y: Sized) -> int:
    """Computes longest common subsequence between X and Y: https://en.wikipedia.org/wiki/Longest_common_subsequence

    Args:
        X (list or str): A string or list of comparable entities.
        Y (list or str): A string or list of comparable entities.

    Returns:
        int: The longest common subsequence between X and Y.
    """
    # find the length of the strings
    m = len(X)
    n = len(Y)
    # declaring the array for storing the dp values
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1]+1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]


def compute_avg_distance(sequence_list: list[Sized], distance_func: Callable, *args, **kwargs) -> np.floating:
    """
    Computes the average distance between all pairs of sequences in sequence_list. The distance is computed

    Args:
        sequence_list (list[Sized]): List of sequences (strings or lists of comparable entities).
        distance_func (Callable): Distance function between two sequences.

    Returns:
        np.floating: The average distance between all pairs of sequences in sequence_list.
    """
    dist_list = []
    for i, seq_1 in enumerate(sequence_list):
        for seq_2 in sequence_list[i + 1:]:
            dist_list.append(distance_func(seq_1, seq_2, *args, **kwargs))
    return np.mean(dist_list)


def average_over_repeats(repeat_list: list[list[Sized]], function: Callable, *args, **kwargs) -> tuple[np.floating, np.floating]:
    """
    Computes the average distance between all pairs of sequences in sequence_list. The distance is computed for each list of lists of repetitions
    for statistical significance. The structure of the input should be repetitions[sequence_list[sequence]].

    Args:
        repeat_list (list[list[Sized]]): List of lists of sequences (strings or lists of comparable entities).
        function (Callable): Function to be called for each sequence pair.

    Returns:
        tuple[np.floating, np.floating]: The average and standard deviation of the distance function. Averaged over all repetitions.
    """

    res = [function(rep, *args, **kwargs) for rep in repeat_list]
    return np.mean(res), np.std(res)


def average_over_repeats_pooled(POOL: mp.Pool, repeat_list: list[list[Sized]], function: Callable, *args, **kwargs) -> tuple[np.floating, np.floating]:
    """Much like average_over_repeats but with multiprocessing. Each repetition is computed in parallel.
    Could be faster in some cases but beware of the additional overhead and memory usage.

    Args:
        POOL (mp.Pool): _multiprocessing.Pool_
        repeat_list (list[list[Sized]]): List of lists of sequences (strings or lists of comparable entities).
        function (Callable): Function to be called for each sequence pair.

    Returns:
        tuple[np.floating, np.floating]: The average and standard deviation of the distance function.
    """
    # res = [function(rep, *args, **kwargs) for rep in repeat_list]
    res = POOL.map(partial(function, *args, **kwargs), repeat_list)
    return np.mean(res), np.std(res)
