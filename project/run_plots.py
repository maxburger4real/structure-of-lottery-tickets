from common import tracking
from run_plot import plot

def main():
    first = '2023_10_09_205114'
    last = '2023_10_09_214342'
    model_class = 'common.architectures.SimpleMLP_4_20_20_2'
    path = tracking.persistance_path / model_class 
    selection = [dir for dir in path.iterdir() if dir.stem >= first and dir.stem <= last]

    for path in selection:
        plot(path)

if __name__ == '__main__':
    main()