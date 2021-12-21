# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:36:26 2021

@author: Andrea
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=lighten_color(color, 1.3))
    plt.setp(bp['fliers'], color=color)  

df_ant = pd.read_csv(f'exp_res_hw_dimensioning_parallel_ANTICIPATE.csv')
df_ant['combination'] = df_ant['time_less_than']
df_ant['combination_id'] = range(len(df_ant))
df_ant = df_ant[df_ant[f'y_nScenarios'] != 'no_sol']
df_ant[f'y_param'] = pd.to_numeric(df_ant[f'y_nScenarios'])
df_ant['y_memAvg(MB)'] = pd.to_numeric(df_ant['y_memAvg(MB)'])
df_ant['n_cores'] = pd.to_numeric(df_ant['n_cores'])

df_cont = pd.read_csv(f'exp_res_hw_dimensioning_parallel_CONTINGENCY.csv')
df_cont['combination'] = df_cont['time_less_than']
df_cont['combination_id'] = range(len(df_cont))
df_cont = df_cont[df_cont[f'y_nTraces'] != 'no_sol']
df_cont[f'y_param'] = pd.to_numeric(df_cont[f'y_nTraces'])
df_cont['y_memAvg(MB)'] = pd.to_numeric(df_cont['y_memAvg(MB)'])
df_cont['n_cores'] = pd.to_numeric(df_cont['n_cores'])

df_maxes = pd.concat([df_ant, df_cont]).max()
df_mins = pd.concat([df_ant, df_cont]).min()
print(df_mins)

for alg, param in zip(['ANTICIPATE', 'CONTINGENCY'], ['nScenarios', 'nTraces']):
    df_ant = pd.read_csv(f'exp_res_hw_dimensioning_parallel_{alg}.csv')
    df_ant['combination'] = df_ant['time_less_than']
    df_ant['combination_id'] = range(len(df_ant))
    
    df_ant = df_ant[df_ant[f'y_{param}'] != 'no_sol']
    df_ant[f'y_{param}'] = pd.to_numeric(df_ant[f'y_{param}'])
    df_ant['y_memAvg(MB)'] = pd.to_numeric(df_ant['y_memAvg(MB)'])
    df_ant['n_cores'] = pd.to_numeric(df_ant['n_cores'])
    df_grouped = df_ant.groupby('combination').mean().reset_index().sort_values(['combination_id'])
    row = df_grouped.iloc[0]

    df_grouped = df_grouped.shift(periods=-1, fill_value=0)
    df_grouped.iloc[-1] = row

    figure, axis = plt.subplots(3, 1, figsize=(13,7))
    figure.figsize=(15,6)
    figure.dpi=200
    data = []
    for index, row in df_grouped.iterrows():
        sub_df = df_ant[df_ant['combination'] == row['combination']]
        data.append(sub_df['y_memAvg(MB)'].to_list()) #, float(row['y_memAvg(MB)'])
    
    bpl = axis[0].boxplot(data, flierprops=dict(markerfacecolor='#D7191C', marker='o'))
    print(len(data[0]))
    
    
    data = []
    for index, row in df_grouped.iterrows():
        sub_df = df_ant[df_ant['combination'] == row['combination']]
        data.append(sub_df[f'y_{param}'].to_list()) #, float(row['y_memAvg(MB)'])   

    bpr = axis[1].boxplot(data, flierprops=dict(markerfacecolor='#2C7BB6', marker='o'))
    
    data = []
    for index, row in df_grouped.iterrows():
        sub_df = df_ant[df_ant['combination'] == row['combination']]
        data.append(sub_df[f'n_cores'].to_list()) #, float(row['y_memAvg(MB)'])   

    bcr = axis[2].boxplot(data, flierprops=dict(markerfacecolor='#4DBF3E', marker='o'))
    
    set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
    set_box_color(bpr, '#2C7BB6')
    set_box_color(bcr, '#4DBF3E')
    plt.plot([], c='#D7191C', label='Memory')
    plt.plot([], c='#2C7BB6', label=f'{param}')
    plt.plot([], c='#4DBF3E', label=f'#CPU Cores')
    plt.legend()
    
    ticks = df_grouped['combination'].to_list()
    axis[0].set_xticklabels(ticks)
    axis[1].set_xticklabels(ticks)
    axis[2].set_xticklabels(ticks)
    axis[0].title.set_text(f'{alg}')
    plt.subplots_adjust(hspace=0.2)
    #axis[1].title.set_text('nScenarios')
    plt.xlabel('Time Bound')
    axis[0].set_ylabel('Memory')
    axis[1].set_ylabel(f'{param}')
    axis[2].set_ylabel('nCores')
    #plt.ylabel(f'{param}')
    originallimits = []
    originallimits.append(axis[0].get_ylim())
    originallimits.append(axis[1].get_ylim())
    originallimits.append(axis[2].get_ylim())
    print(originallimits)
    plt.savefig(f'{alg}_boxplot.png')
    axis[0].set_ylim([0, df_ant.max()['y_memAvg(MB)'] + df_ant.max()['y_memAvg(MB)']*0.10])
    axis[1].set_ylim([0, df_ant.max()[f'y_{param}'] + df_ant.max()[f'y_{param}']*0.10])
    axis[2].set_ylim([0, df_ant.max()['n_cores'] + df_ant.max()['n_cores']*0.10])
    plt.savefig(f'{alg}_boxplot_scaled_maxCurAlg.png')
    axis[0].set_ylim([0, df_maxes['y_memAvg(MB)'] + df_maxes['y_memAvg(MB)']*0.10])
    axis[1].set_ylim([0, df_maxes['y_param'] + df_maxes['y_param']*0.10])
    axis[2].set_ylim([0, df_maxes['n_cores'] + df_maxes['n_cores']*0.10])
    plt.savefig(f'{alg}_boxplot_scaled_maxGlobal.png')
    plt.show()

    plt.clf()
    axis[0].clear()
    axis[1].clear()
    
for alg, param in zip(['ANTICIPATE', 'CONTINGENCY'], ['nScenarios', 'nTraces']):
    df_ant = pd.read_csv(f'exp_res_hw_dimensioning_parallel_{alg}.csv')
    df_ant['combination'] = df_ant['time_less_than']
    df_ant['combination_id'] = range(len(df_ant))
    
    df_ant = df_ant[df_ant[f'y_{param}'] != 'no_sol']
    df_ant[f'y_{param}'] = pd.to_numeric(df_ant[f'y_{param}'])
    df_ant['y_memAvg(MB)'] = pd.to_numeric(df_ant['y_memAvg(MB)'])
    df_ant['y_memAvg(MB)'] = df_ant['y_memAvg(MB)'] / df_ant['y_memAvg(MB)'].min()
    df_ant['n_cores'] = pd.to_numeric(df_ant['n_cores'])
    df_grouped = df_ant.groupby('combination').mean().reset_index().sort_values(['combination_id'])
    row = df_grouped.iloc[0]

    df_grouped = df_grouped.shift(periods=-1, fill_value=0)
    df_grouped.iloc[-1] = row

    figure, axis = plt.subplots(3, 1, figsize=(13,7))
    figure.figsize=(15,6)
    figure.dpi=200
    data = []
    for index, row in df_grouped.iterrows():
        sub_df = df_ant[df_ant['combination'] == row['combination']]
        data.append(sub_df['y_memAvg(MB)'].to_list()) #, float(row['y_memAvg(MB)'])
    
    bpl = axis[0].boxplot(data, flierprops=dict(markerfacecolor='#D7191C', marker='o'))
    print(len(data[0]))
    
    data = []
    for index, row in df_grouped.iterrows():
        sub_df = df_ant[df_ant['combination'] == row['combination']]
        data.append(sub_df[f'y_{param}'].to_list()) #, float(row['y_memAvg(MB)'])   

    bpr = axis[1].boxplot(data, flierprops=dict(markerfacecolor='#2C7BB6', marker='o'))
    
    data = []
    for index, row in df_grouped.iterrows():
        sub_df = df_ant[df_ant['combination'] == row['combination']]
        data.append(sub_df[f'n_cores'].to_list()) #, float(row['y_memAvg(MB)'])   

    bcr = axis[2].boxplot(data, flierprops=dict(markerfacecolor='#4DBF3E', marker='o'))
    
    set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
    set_box_color(bpr, '#2C7BB6')
    set_box_color(bcr, '#4DBF3E')
    plt.plot([], c='#D7191C', label='Normalized Memory')
    plt.plot([], c='#2C7BB6', label=f'{param}')
    plt.plot([], c='#4DBF3E', label=f'#CPU Cores')
    plt.legend()
    
    ticks = df_grouped['combination'].to_list()
    axis[0].set_xticklabels(ticks)
    axis[1].set_xticklabels(ticks)
    axis[2].set_xticklabels(ticks)
    axis[0].title.set_text(f'{alg}')
    plt.subplots_adjust(hspace=0.2)
    #axis[1].title.set_text('nScenarios')
    plt.xlabel('Time Bound')
    axis[0].set_ylabel('Memory')
    axis[1].set_ylabel(f'{param}')
    axis[2].set_ylabel('nCores')
    #plt.ylabel(f'{param}')

    plt.savefig(f'{alg}_boxplot_memNormalized.png')
    plt.show()

    plt.clf()
    axis[0].clear()
    axis[1].clear()
