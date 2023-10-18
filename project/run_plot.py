from common import plotting
from common import tracking

def plot(path):
    plotting.plot_checkpoints(path)

def main():
    model_class = 'BioMLP_4_20_20_2'
    run_id = "qv2wt6or"

    model_class = 'SimpleMLP_4_20_20_2'
    run_id = "nftxsj0c"
    path = tracking.persistance_path / model_class / run_id
    plot(path)

if __name__ == '__main__':
    main()