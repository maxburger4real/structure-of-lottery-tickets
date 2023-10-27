import wandb
from common import tracking
from common import plotting
from common.constants import *
 
entity = ENTITY
project = PROJECT

def main():
    
    # L1 test
    sweep_id = '54yvf22u'
    
    # L1 ADAM 
    sweep_id = 'vobzd18r'

    sweep_id = None

    # classic BIMT 
    #sweep_id = 'nsm9pspb'
    if sweep_id is not None:
        api = wandb.Api()
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        run_ids = [run.id for run in sweep.runs]

    run_ids = [
        'evmykexg',
        #'4shs3f8q',
        #'pgl0pqdh',
        #'oqhn7xq5', 
        #'gcme9bvs'
        ]

    for name in run_ids:
        path = tracking.persistance_path / name
        plotting.plot_checkpoints(path)

if __name__ == '__main__':
    main()