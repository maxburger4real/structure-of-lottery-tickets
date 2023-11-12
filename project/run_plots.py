import wandb
from common.persistance import persistance_path
from common.plotting import plot_checkpoints
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
        'xi8wprwh',
        'k90pihpr',
        '67y2zskn',
        'x2gkwc1b',
        'czbqa4ve',
        #'hezzefi6',
        #'uj4oim94',
        #'drtwf1qd',
        #'evmykexg',
        #'4shs3f8q',
        #'pgl0pqdh',
        #'oqhn7xq5', 
        #'gcme9bvs'
        ]

    for name in run_ids:
        plot_checkpoints(persistance_path / name)

if __name__ == '__main__':
    main()