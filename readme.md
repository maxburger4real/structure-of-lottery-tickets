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
