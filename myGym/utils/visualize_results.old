import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import yaml
import difflib
from difflib import SequenceMatcher
import re

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pths", default=['./trained_models/cluster/reach_kuka_absolute_gt_ppo2/',
                                          './trained_models/cluster/reach_kuka_absolute_gt_ppo/',
                                          './trained_models/cluster/reach_kuka_absolute_gt_a2c/',
                                          './trained_models/cluster/reach_kuka_absolute_gt_acktr/',
                                          './trained_models/cluster/reach_kuka_joints_gt_trpo/'], nargs='*')
    parser.add_argument("-pth", default=['./trained_models/pnrppo2'])
    args = parser.parse_args()

    return args


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
    return {key: value for key, _, value, _ in re.findall(
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
        if not all([k==keys[0] for k in keys]):
            raise Exception("Oops, error! Not all keys were the same. Lines in the configs must be mixed up.")
        if all([v==vals[0] for v in vals]):
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
    all_same = [all([v==line[0] for v in line]) for line in all_vals]
    return {record[0][0]: (record[0][1] if all_same[i] else [v for k, v in record]) for i, record in enumerate(zip(*[c.items() for c in dict_list]))}, all_same

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
    return {k: v for i, (k, v) in enumerate(diff_dict.items()) if not all_same[i]}, {k: v for i, (k, v) in enumerate(diff_dict.items()) if all_same[i]}

def main():
    args = get_arguments()
    #subdirs = [x[0] for x in os.walk(str(args.pth[0]))][1:]
    root, dirs, files = next(os.walk(str(args.pth[0])))
    plt.rcParams.update({'font.size': 12})
    fig, axs = plt.subplots(4, sharex=True, sharey=False, gridspec_kw={'hspace': 0})
    colors = ['red','green','blue','yellow','magenta','cyan','black','grey','brown','gold','limegreen','silver','aquamarine','olive','hotpink','salmon']
    configs=[]
    for idx,model in enumerate(dirs):
        with open(os.path.join(root+"/"+model,"evaluation_results.json")) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        x = df.to_numpy()
        x = x.astype(np.float)


        with open(os.path.join(root+"/"+model,"train.conf"), "r") as f:
            # load individual configs
            cfg_raw = yaml.full_load(f)
            #cfg_crop = (cfg_raw.split(" model_path")[0])
            # and convert them to dictionary
            configs.append(cfgString2Dict(cfg_raw))
        #fig.suptitle(cfg)


        axs[0].plot(x[0],x[3], color=colors[idx], linestyle='solid', linewidth = 3, marker='o', markerfacecolor=colors[idx], markersize=6)
        axs[1].plot(x[0],(x[4]*100), color=colors[idx], linestyle='solid', linewidth = 3, marker='o', markerfacecolor=colors[idx], markersize=6)
        axs[2].plot(x[0],x[5], color=colors[idx], linestyle='solid', linewidth = 3, marker='o', markerfacecolor=colors[idx], markersize=6)
        axs[3].plot(x[0],x[6], color=colors[idx], linestyle='solid', linewidth = 3, marker='o', markerfacecolor=colors[idx], markersize=6)

    # get differences between configs
    diff, same = multiDictDiff_bykey(configs)
    fig.text(0.1,0.94,same, ha="left", va="bottom", size="medium",color="black", wrap=True)
    #l = []
    #for key, value in diff.items():
    #    l.append(value)
    #print(len(l))
    #res = [a+" "+b for a,b in zip(l[0], l[1])]
    #plt.legend(res)
    s = list(diff.values())
    leg = []
    for ix, x in enumerate(s[0]):
        leg.append(" ".join((x, s[0][ix])))


    plt.legend(leg,loc=8)

    ylabels = ['Successful episodes (%)', 'Goal distance (%)', "Mean episode steps","Mean reward"]
    for idx, ax in enumerate(axs):
        ax.label_outer()
        ax.set(ylabel=ylabels[idx])

    plt.xlabel('Training steps')
    plt.show()

if __name__ == "__main__":
    main()
