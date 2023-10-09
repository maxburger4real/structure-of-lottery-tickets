from common import plotting
from common import tracking

def plot(path):
    G = plotting.plot_checkpoints(path)

def main():
    model_class = 'common.architectures.SimpleMLP_4_20_20_2'
    timestamp = "2023_10_09_205114" 
    path = tracking.persistance_path / model_class / timestamp
    plot(path)

if __name__ == '__main__':
    main()