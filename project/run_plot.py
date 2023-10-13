from common import plotting
from common import tracking

def plot(path):
    G = plotting.plot_checkpoints(path)

def main():
    model_class = 'common.architectures.SimpleMLP_4_20_20_2'
    
    # Good
    timestamp = "2023_10_09_230020" 
    timestamp = "2023_10_09_225503" 

    # Mid
    timestamp = "2023_10_09_230155" 


    timestamp = "2023_10_09_224014" 
    # BAD

    path = tracking.persistance_path / model_class / timestamp
    plot(path)

if __name__ == '__main__':
    main()