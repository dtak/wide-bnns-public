# standard library imports
import argparse
import itertools
import os
import sys
import importlib
import time

# package imports
import pandas as pd
import numpy as np

__version__ = '0.1.0'
__author__ = u'Beau Coker'

'''
Script for re-running a script ("subprocess") for different command line argument

Arguments:
--args: text file containing arguments to run (see required syntax below)
--subproc: python script to run (e.g., './experiment.py')
--search_mode: rule for searching over parameters (e.g. gridsearch)
--dir_out: directory of where to store results
--arg_dir_out: name of subprocess argument corresponding to where results are stored (optional)

Notes:
- if arg_dir_out is specified, then results from run i will be stored at "dir_out/i"
- if subprocess returns a dictionary, results will be stored as a dataframe in "dir_out/aggregate_results.pkl",
  where each row corresponds to a single run 
- if search_mode=='grid' then all argument combinations are run 
- if search_mode=='zip' then argument values are "zipped" together. Requires all lengths can divide longest length.

Syntax for argument files (comments # should not be included):
--argument1: 1.0 # can be a scalar...
--argument2: ["a", "b"] # ... or a list
--argument3: [i**2 for i in [1, 2]] # ... or an iterable
--argument4: np.array([1, 2]) # ... or a numpy array
--argument5: # ... or a boolean argument (leave blank if so)
'''

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args', type=str, default='./args.txt')
    parser.add_argument('--subproc', type=str, default='./sub.py')
    parser.add_argument('--search_mode', type=str, default='grid')
    parser.add_argument('--dir_out', type=str, default='./output/')
    parser.add_argument('--arg_dir_out', type=str, default='--dir_out', help='argument name for storing subprocess output (set to None if no output)')
    return parser


def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)

    # import subprocess
    sys.path.append(os.path.dirname(args.subproc))
    module = os.path.split(os.path.splitext(args.subproc)[0])[-1]
    sub = importlib.import_module(module)

    # output directory
    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out)

    # read subprocess argument from file
    args_sub = {}
    with open(args.args) as f: 
        for line in f:
            # argument with value
            (key, val) = line.split(':')
            try:
                args_sub[key] = [str(item) for item in eval(val)]
            except:
                try:
                    args_sub[key] = [str(eval(val))]
                except:
                    args_sub[key] = [val.lstrip().rstrip()]

    # compute all argument combinations 
    if args.search_mode == 'grid':
        argvalcombs = list(itertools.product(*args_sub.values()))
    elif args.search_mode == 'zip':
        val_lengths = [len(v) for v in args_sub.values()]
        if not all([max(val_lengths)%l ==0 for l in val_lengths]):
            raise Exception("number of values for each argument must divide largest number of values for an argument")
        val_reps = [max(val_lengths)//len(v) for v in args_sub.values()]
        argvalcombs = list(zip(*[r*v for r,v in zip(val_reps, args_sub.values())]))
    else:
        raise Exception("search_mode not recognized")

    # create dataframe with each row corresponding to a separate experiment
    argsdf = pd.DataFrame(argvalcombs, columns=[name for name in args_sub.keys()])
    if args.arg_dir_out != "None":
        argsdf[args.arg_dir_out] = [os.path.join(args.dir_out, '%d' % i) for i in range(argsdf.shape[0])]
    argsdf.to_csv(os.path.join(args.dir_out, 'argument_permutations.csv'))

    # loop over all argument combinations
    print('Beginning experiments...')
    results_list = []
    for index, row in argsdf.iterrows():
        
        args_list = [x for y in zip(argsdf.columns.tolist(), row.tolist()) for x in y]
        args_list = list(filter(lambda z: z!='', args_list)) # filter out '' argument values, which are assumed to correspond to boolean arguments
        
        start_time = time.time()
        results = sub.main(args_list)
        runtime = time.time() - start_time

        print('Completed experiment [%d/%d] (time: %.3f seconds)' % (index+1, argsdf.shape[0], runtime))
        
        if results is not None:
            results['runtime_experiment'] = runtime
            results_list.append(results)

            # aggregate and save current results
            resultsdf = pd.concat([argsdf, pd.DataFrame(results_list)], axis=1)
            resultsdf.to_pickle(os.path.join(args.dir_out, 'aggregate_results.pkl'))
    
    # aggregate and save final results
    if len(results_list) > 0:
        resultsdf = pd.concat([argsdf, pd.DataFrame(results_list)], axis=1)
        resultsdf.to_pickle(os.path.join(args.dir_out, 'aggregate_results.pkl'))
        resultsdf.to_csv(os.path.join(args.dir_out, 'aggregate_results.csv')) # also save as csv for easier access

if __name__ == '__main__':
    main()