# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:05:59 2020
​
@author: rados
"""

import yaml
import os
import re

pths = [r'./trained_models/reach_kuka_joints_gt_ppo2/',
        r'./trained_models/reach_kuka_step_gt_ppo2/',
        r'./trained_models/reach_kuka_absolute_gt_ppo2/',
        r'./trained_models/reach_panda_absolute_gt_ppo2/']

# pattern to extract dictionary from a string
# pattern = re.compile(r"(\w+\s?)(=\s?)([^=]+)( (?=\w+\s?=)|$)")


def cfgString2Dict(cfg_raw):
    """
    Converts a string with configuration parameters separated by a space
    into a dictionary.
​
    Parameters
    ----------
    cfg_raw : str
​
    Returns
    -------
    dict
    """
    return {key.strip(): value.strip() for key, _, value, _ in re.findall(
        r"(\w+\s?)(=\s?)([^=]+)( (?=\w+\s?=)|$)", cfg_raw)}


def dict2cfgString(dictionary, separator="\n", assigner="="):
    """
    Converts a dictionary into a string
​
    Parameters
    ----------
    dictionary : dict
        The dictionary to be transformed.
    separator : str, optional
        The character to be used to separate individual
        entries. The default is "\n".
    assigner: str, optional
        The character to represent the assignment from the key
        to the value. The default is "=".
​
    Returns
    -------
    str
    """
    return "{}".format(separator).join([f"{k}{assigner}{v}" for k, v in dictionary.items()])


def multiDictDiff_byline(dict_list):
    """
    Compares multiple dictionaries in a list, expects the same order of keys
​
    Parameters
    ----------
    dict_list : list of dict
​
    Raises
    ------
    Exception
        Throws an exception if the order of keys is not the same in all dictionaries
​
    Returns
    -------
    different_items : dict
        A dictionary where the keys are the keys from original dictionaries
        that were not the same in all provided dictionaries.
        The values are lists of values for the individual dictionaries.
        Order of the values in the lists is the same as the order of dictionaries
        in the input list.
    same_items : dict
        A dictionary with keys and values that were the same for all
        provided dictionaries. Single value per key (i.e. not a list
                                                     as in previous output).
​
    """
    different_items = {}
    same_items = {}
    for record in zip(*[c.items() for c in dict_list]):
        vals = [v for k, v in record]
        keys = [k for k, v in record]
        if not all([k == keys[0] for k in keys]):
            raise Exception("Oops, error! Not all keys were the same. Lines in the configs must be mixed up.")
        if all([v == vals[0] for v in vals]):
            same_items[keys[0]] = vals[0]
        else:
            different_items[keys[0]] = vals

    return different_items, same_items


def multiDictDiff_scary(dict_list):
    """
    Compares a list of dictionaries. Expects all dictionaries to have the same keys.
​
    Parameters
    ----------
    dict_list : list of dict
​
    Returns
    -------
    dict
        A dictionary where for each key the value is either the original
        value from the provided dicts, iff it was the same in all dicts.
        Or, the value for the specific key is a list of values for
        individual dictionaries, iff it was not the same in all of them.
    all_same : list
        A list that shows for each entry whether it was the same or not.
​
    """
    all_vals = [[d[k] for d in dict_list] for k in dict_list[0].keys()]
    all_same = [all([v == line[0] for v in line]) for line in all_vals]
    return {record[0][0]: (record[0][1] if all_same[i] else [v for k, v in record]) for i, record in
            enumerate(zip(*[c.items() for c in dict_list]))}, all_same


def multiDictDiff_bykey(dict_list):
    """
    Compares a list of dictionaries. Expects all dictionaries to have the same keys.
​
    Parameters
    ----------
    dict_list : list of dict
​
    Returns
    -------
    different_items : dict
        A dictionary where the keys are the keys from original dictionaries
        that were not the same in all provided dictionaries.
        The values are lists of values for the individual dictionaries.
        Order of the values in the lists is the same as the order of dictionaries
        in the input list.
    same_items : dict
        A dictionary with keys and values that were the same for all
        provided dictionaries. Single value per key (i.e. not a list
                                                     as in previous output).
    """
    diff_dict, all_same = multiDictDiff_scary(dict_list)
    return {k: v for i, (k, v) in enumerate(diff_dict.items()) if not all_same[i]}, {k: v for i, (k, v) in
                                                                                     enumerate(diff_dict.items()) if
                                                                                     all_same[i]}


# %% Example usage
# configs = []
# for p in pths:
#     with open(os.path.join(p, "train.conf"), "r") as f:
#         # load individual configs
#         cfg_raw = yaml.full_load(f)
#         # and convert them to dictionary
#         configs.append(cfgString2Dict(cfg_raw))
#
# # get differences between configs
# diff, same = multiDictDiff_bykey(configs)