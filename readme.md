# Inside the structure of Lottery tickets


# CHECKOUT THIS BRANCH  and RUN 
python run_experiment.py -r configs/runs/_11_imp_moons_good.py

this creates network splitting.


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

# Do it with VSCode Remote Extension
Add remote host 