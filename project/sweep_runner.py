
import wandb
from common.tracking import PROJECT
import runner
from training_pipelines.imp import BEST_EVAL
sweep_config = {
    'project': PROJECT,
    'method': 'random', # random, grid, bayes
    'name': 'my-first-real-sweep',
    'metric': {
        'name': BEST_EVAL,
        'goal': 'minimize' 
    },
    'parameters':{

        "training_epochs": {"values": [1000, 1500]},
        "lr": {"max": 0.07, "min": 0.0001},
        }
    }

def main():
    sweep_id = wandb.sweep(sweep_config, project=PROJECT)
    wandb.agent(sweep_id, runner.main, count=1)

if __name__ == "__main__":
    main()