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

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

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


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pth", default='./trained_models/pnp4nd')
    parser.add_argument("-task", default='pnrmulti')
    parser.add_argument("-robot", default=["kuka"], nargs='*')
    parser.add_argument("-common", default='pnprot_table_kuka_joints')
    parser.add_argument("-algo", default=["multi","ppo2","ppo","acktr","sac","ddpg","a2c","acer","trpo"], nargs='*')
    parser.add_argument("-xlabel", type=int, default=1)
    args = parser.parse_args()

    return args


def main():
    args = get_arguments()
    #subdirs = [x[0] for x in os.walk(str(args.pth[0]))][1:]
    root, dirs, files = next(os.walk(str(args.pth)))
    dirs.sort(key=natural_keys)
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(8,3))
    #fig, axs = plt.subplots(4, sharex=True, sharey=False, gridspec_kw={'hspace': 0})
    colors = ['red','green','blue','yellow','magenta','cyan','black','grey','brown','gold','limegreen','silver','aquamarine','olive','hotpink','salmon']
    configs=[]
    success = []
    min_steps = 100
    for idx, file in enumerate(dirs):
        evaluation_exists = False
        try:
            with open(os.path.join(args.pth,file, "evaluation_results.json")) as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            x = df.to_numpy()
            
            try:
                x = np.delete(x, [10,11,12], axis=0)
            except:
                print("No multistep reward data")
            
            x = x.astype(float)
            with open(os.path.join(args.pth,file, "train.json"), "r") as f:
                # load individual configs
                cfg_raw = yaml.full_load(f)
                
            
            success.append(x[3])
            configs.append(cfg_raw)
            if len(x[0])< min_steps:
                min_steps = len(x[0])

            steps = x[0][:min_steps]
            print("{} datapoints in folder {}".format(len(x[3]),file))
            
        except:
            print("0 dataponits in folder:{} ".format(file))
        
    # get differences between configs
    diff, same = multiDictDiff_bykey(configs)
    leg = []
    for d in range(len(success)):
        success[d] = np.delete(success[d],np.s_[min_steps:])
    if 'algo' in diff.keys() and len(args.algo)>1:
        index = [[] for _ in range(len(args.algo))]
        for i, algo in enumerate(args.algo):
            for j, diffalgo in enumerate(diff['algo']):
                if algo == diffalgo:
                    index[i].append(j)
            if len(index[i])>0:
                meanvalue = np.mean(np.take(success,index[i],0),0)
                variance =  np.std(np.take(success,index[i],0),0)
                plt.plot(steps,meanvalue, color=colors[i], linestyle='solid', linewidth = 3, marker='o', markerfacecolor=colors[i], markersize=4) 
                plt.fill_between(steps, np.mean(np.take(success,index[i],0),0)-np.std(np.take(success,index[i],0),0),np.mean(np.take(success,index[i],0),0)+np.std(np.take(success,index[i],0),0), color=colors[i], alpha=0.2) 
                #plt.show()
                leg.append(algo)
                print(algo)
                print(meanvalue[-1])
                print(variance[-1]) 
           
    elif 'robot' in diff.keys() and len(args.robot)>1:
        index = [[] for _ in range(len(args.robot))]
        for i, robot in enumerate(args.robot):
            for j, diffrobot in enumerate(diff['robot']):
                if robot == diffrobot:
                    index[i].append(j)
            if len(index[i])>0:
                plt.plot(x[0],np.mean(np.take(success,index[i],0),0), color=colors[i], linestyle='solid', linewidth = 3, marker='o', markerfacecolor=colors[i], markersize=6) 
                plt.fill_between(x[0], np.mean(np.take(success,index[i],0),0)-np.std(np.take(success,index[i],0),0),np.mean(np.take(success,index[i],0),0)+np.std(np.take(success,index[i],0),0), color=colors[i], alpha=0.2) 
                #plt.show()
                leg.append(robot)
    else:
        print("No data to visualize")        
        
    
    #s = list(diff.values())
    #leg = []
    #for ix, x in enumerate(s[0]):
    #    leg.append(x)

    # plt.label_outer()
    plt.ylabel('Successful episodes {}(%)'.format(args.task))
    
    if args.xlabel:
    	plt.xlabel('Training steps')
    	plt.legend(leg,loc=2)
    #plt.ylim(-3, 103)
    plt.tight_layout()
    plt.savefig(root+"-averaged.png")
    print(root)
    #plt.show()
if __name__ == "__main__":
    main()
