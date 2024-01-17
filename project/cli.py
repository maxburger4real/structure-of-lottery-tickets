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

    runner.run(
        experiment_file=args.experiment,
        run_file=args.runconfig,
        sweep_file=args.sweepconfig,
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
