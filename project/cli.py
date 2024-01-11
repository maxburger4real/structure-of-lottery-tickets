"""
USAGE
-d wandb disable
python project/run_experiment.py -d -r project/configs/runs/_000_debug.py
python project/run_experiment.py -d -r project/configs/runs/_000_debug.py
python project/run_experiment.py -d -r project/configs/runs/_000_debug.py -s project/configs/sweeps/_01_compare_with_multiple_seeds
"""

import argparse
import argcomplete

import runner

from common.constants import *
from common.config import import_config


def main():

    parser = argparse.ArgumentParser()
    argcomplete.autocomplete(parser)  # Enable autocompletion with argcomplete

    # experiments  (run + sweep config in a folder)
    parser.add_argument('-e', '--experiment', help='run experiment. run config and sweep config in a single file')
    parser.add_argument('-r', '--runconfig', help='run config file')
    parser.add_argument('-s', '--sweepconfig', help='sweep config file')
    parser.add_argument('--count', help="The maximum number of runs for this sweep")

    # debug - disbale wandb logging to speedup
    parser.add_argument('-d', '--disable', action='store_true', help="Disable WANDB mode.")
    
    # THE ARGHSSSSSS
    args = parser.parse_args()

    if args.experiment is not None:
        runconfig, sweepconfig = import_config(args.experiment)
    elif args.runconfig is not None:
        runconfig, _ = import_config(args.runconfig)
        _ ,sweepconfig = import_config(args.sweepconfig)
    else:
        raise ValueError('Must provide either experiment or runconfig.')
        
    if runconfig is None: raise ValueError('No valid runconfig found.')
    
    runner.run(
        runconfig, 
        sweepconfig, 
        mode=__parse_mode(args),
        count=__parse_count(args)
    )

def __parse_mode(args):

    if args.disable:
        return 'disabled'
    
    return 'online'

def __parse_count(args):
    count = None
    if args.count is not None:
        if args.count.isdigit(): count = int(args.count)
        else: print(f'Ignoring args.count {args.count}')
    return count

if __name__ == "__main__":
    main()
