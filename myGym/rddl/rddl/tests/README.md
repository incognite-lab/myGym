# How to evaluate sampling

## Important files  

### test_world.py  

Runs world sampling experiments. Settings are at the beginning of the `test_sampling_eff_multi` function.
The generated sequences are (optionally) stored to a csv file.

### show_test_world_results.py  

Shows graphs of metrics from `test_world.py` result.

### compute_max_seq.py  

Computes all viable sequences. Length of sequence can be set at the beginning of the script. Output is stored into a csv file.

### compute_all_seq.py  

Computes all sequences of given length (specified at the beginning of the script). These are all, not only possible sequences. Output is stored into a csv file.

### compute_optimal_positions.py  

Computes optimal positions of nodes with weighted edges in a graph. This is done over all possible sequences from `compute_all_seq.py`.

### viz_dist_histogram.py  

Shows histogram of distances between sequences for all, viable and optionally sampled sequences.

### viz_pair_distance.py  

Shows graph plot of all, viable and (optionally) sampled sequences, using optimized positions of nodes/sequences.

## How to  

Typically scenario: I want to sample some sequences, show their distances compared to all/viable sequences. To do this:

1) Run `test_world.py` with the desired settings (e.g. length of sequence). This will create a csv with the generated sequences. Prefix of the file will be `gseq_`.  
2) Generate all and viable sequences, using the `compute_all_seq.py` and `compute_max_seq.py` respectively. This will create csv files prefixed with `all-sequences-` and `sequences-`. It will contain the length of the sequence, which can be set at the beginning of the respective scripts.  
3) Generate optimal graph positions using the `compute_optimal_positions.py` file. Set the sequence length variable in the script to load the correct csv file.  
4) Visualize the results either using `viz_dist_histogram.py` or `viz_pair_distance.py` scripts. Set the sequence length at the beginning to load the correct data. Optionally, set the variable `compare_to` to the name of the csv file with sampled sequences (from `test_world.py`), prefixed with `gseq_`. You need to set the full path to the file.  

Note: there might be some path issues when loading the csv files.
