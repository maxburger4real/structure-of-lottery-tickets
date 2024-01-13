
import wandb
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__WANDB_PROJECT = "mxmn/concat_moons/"


splitters = {
    'split only':'#68B684',
    'split + degrade':'#508B65' 
}
nonsplitters = {
    'still connected':'#B84A62',
    'degrade only': '#FFC107',#'#AC7B84',
}


def runs_from_sweeps(sweep_ids):
    api = wandb.Api(timeout=60)
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

def make_df_from_history(histories_and_config, key, invert, drop_first):
    rows = []
    for h,_ in histories_and_config:

        df = h[key]
        if drop_first:
            # 0-th entry is often always None, no pruning happens
            df.drop(0, inplace=True)

        if invert:
            # invert to align where p-params is equal
            df_inverted = df.iloc[::-1].reset_index(drop=True)
            df_inverted.index = df.index
            df = df_inverted

        rows.append(df)

    return pd.DataFrame(rows).reset_index(drop=True)


# helpers
def make_group(group_df, source_df, source_keys):

    assert len(group_df.columns) == 1, 'only one column allowed'
    group_attr = group_df.columns[0]

    group = pd.concat(
        [group_df] + [source_df[k] for k in source_keys], 
        axis=1
    ).groupby(group_attr)

    return group

def mean_std_from_group(group, key):
    return pd.DataFrame(
        data={
            'xticks' : group.mean().reset_index().index,
            'xticklabels' : group.mean().index,
            'x' : group.mean().index,
            'y' : group.mean()[key],
            'yerr' : group.agg(np.std, ddof=0)[key],
        }
    )

def errorbar_from_df(df, **kwargs):
    plt.errorbar(
        x=df['x'], 
        y=df['y'], 
        yerr=df['yerr'],
        **kwargs
    )
    plt.xticks(df['x'])
    plt.legend()


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