# Inside the structure of Lottery tickets

### conda - setup

First, create an environment and activate it.
The following installs were initially done.
```zsh
% conda --version  
conda 23.5.0

% conda create --prefix=venv python=3.11
% conda activate ./venv                 
% conda install pytorch -c pytorch      
% conda install -c anaconda ipykernel    
% conda install -c conda-forge numpy      
% conda install openblas                    
```

to export the environment, use:
```zsh
% conda list -e > requirements.txt
```

### remote 
https://github.com/maxburger4real/structure-of-lottery-tickets


On the GPU server:

Confgire the shell like they write in the readme. Source to the userenv.

Then conda is available 

Create a conda environment

    conda create --preifx=venv python=3.11

