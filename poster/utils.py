
import wandb
from collections import defaultdict
import numpy as np
import pandas as pd

__WANDB_PROJECT = "mxmn/concat_moons/"

def runs_from_sweeps(sweep_ids):
    api = wandb.Api()
    base = __WANDB_PROJECT
    runs = []
    for id in sweep_ids:
        for run in api.sweep(base + id).runs:
            runs.append(run)
    return runs

def load_histories_and_configs(sweep_ids):
    runs = runs_from_sweeps(sweep_ids)
    return [(run.history(), run.config) for run in runs]

def is_split(history):
    potential = history['untapped-potential']
    return (potential == 0).any()

def is_degrade(history):
    potential = history['untapped-potential']
    return (potential < 0).any()

def split_range_from_history(history):
    idc = np.where(history['untapped-potential'].values == 0)[0]
    if len(idc) > 0:
        begin, end = int(idc[0]), int(idc[-1])
    else: 
        begin, end = None, None

    return begin, end

# predefined dataframes
def make_split_df(histories_and_config):
    data = [
        {
            'split only': is_split(h) and not is_degrade(h),
            'split + degrade':is_split(h) and is_degrade(h),
            'degrade only':not is_split(h) and is_degrade(h),
            'still connected':not is_split(h) and not is_degrade(h)
        }
        for h, _ in histories_and_config
    ]
    return pd.DataFrame(data)

def make_df(histories_and_config, *keys):
    data = []
    for h, c in histories_and_config:
        row = {}
        for key in keys:
            if key in h: raise ValueError('Only works for config keys')
            elif key in c: value = c[key]
            row[key] = value

        data.append(row)

    return pd.DataFrame(data)

def make_splitrange_df(histories_and_config):
    data=[]
    for h, _ in histories_and_config:
        first, last = split_range_from_history(h)
        data.append({
            'first_split': first,
            'last_split': last,
        })
    return pd.DataFrame(data)

def make_dfs_at_splitrange(hc, df):

    data_first, data_last=[], []

    # insert a np.nan row when it didnt split
    dummy = pd.Series([np.nan]*len(hc[0][0].columns))

    # loop over splitrange df
    for i, (first, last) in df.iterrows():
        history, _ = hc[i]
        data_first.append(history.iloc[int(first)] if not np.isnan(first) else dummy)
        data_last.append(history.iloc[int(last)] if not np.isnan(last) else dummy)
    
    return pd.DataFrame(data_first, index=df.index), pd.DataFrame(data_last, index=df.index)


# plotting format
def set_style(axes):
    for ax in axes:
        ax.set_title('')
        ax.grid('on', color='#CCCCCC', linestyle=':', axis='y')
        ax.spines['left'].set_color((1.,1.,1.))
        ax.spines['right'].set_color((1.,1.,1.))
        ax.spines['top'].set_color((1.0,1.0,1.0))
        ax.spines['bottom'].set_color([0.8]*3)

def load_data(runs):
    x_axis = set()
    data = defaultdict(lambda: defaultdict(int))
    for run in runs:
        level = run.config['pruning_levels']
        x_axis.add(level)

        key='untapped-potential'
        hist = run.history(keys=[key])
        potential = hist[key]
        degraded = (potential < 0).any()
        split = (potential == 0).any()

        data['s'][level] += 1 if (split and not degraded) else 0
        data['sd'][level] += 1 if (degraded and split) else 0
        data['d'][level] += 1 if (degraded and not split) else 0
        data['no'][level] += 1 if ((not degraded) and (not split)) else 0
        data['pr'][level] = run.config['pruning_rate']

    return data, x_axis