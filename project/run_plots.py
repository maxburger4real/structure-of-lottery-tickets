import wandb
from common import tracking
from common import plotting
from common.tracking import PROJECT
 
entity = 'mxmn'
project = PROJECT

def main():
    
    sweep_id = 'c5cm3766'
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    run_ids = [run.id for run in sweep.runs]
    
    for name in run_ids:
        path = tracking.persistance_path / name
        plotting.plot_checkpoints(path)

if __name__ == '__main__':
    main()