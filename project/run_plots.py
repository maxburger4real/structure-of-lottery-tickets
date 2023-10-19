from common import tracking
from common import plotting

def plot(path):
    plotting.plot_checkpoints(path)
 
def main():

    models = {
        'SimpleMLP_4_20_20_2' : ["ssoy8hp4", ],
        'BioMLP_4_20_20_2' : ["qv2wt6or"]
    }

    for model_class, runs in models.items(): 
        for name in runs:
            path = tracking.persistance_path / model_class / name
            plot(path)

if __name__ == '__main__':
    main()