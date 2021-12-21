# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:36:26 2021

@author: Andrea
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

def change_id(row):
    if row['combination'] == 'no':
        return 900
    else:
        return int(row['combination'])

datalist = []
for alg, param in zip(['ANTICIPATE', 'CONTINGENCY'], ['nScenarios', 'nTraces']):
    df_ant = pd.read_csv(f'exp_res_hw_dimensioning_parallel_{alg}.csv')
    df_ant['combination'] = df_ant['time_less_than']
    #df_ant['combination'] = df_ant.apply(lambda x: change_id(x), axis=1)
    df_ant['index_'] = len(df_ant)
    df_ant['combination_id'] = df_ant.apply(lambda x: change_id(x), axis=1).rank(method='dense')
    df_ant = df_ant[df_ant[f'y_{param}'] != 'no_sol']
    df_ant[f'y_{param}'] = pd.to_numeric(df_ant[f'y_{param}'])
    df_ant['y_memAvg(MB)'] = pd.to_numeric(df_ant['y_memAvg(MB)'])
    df_ant['n_cores'] = pd.to_numeric(df_ant['n_cores'])
    
    print(df_ant['combination_id'] )
    df_grouped = df_ant.groupby('combination').mean().reset_index().sort_values(['index_'])

    row = df_grouped.iloc[0]

    df_grouped = df_grouped.shift(periods=-1, fill_value=0)
    df_grouped.iloc[-1] = row

    fig = plt.figure(figsize=(9,6))
    fig.dpi=200

    new_cool = matplotlib.colors.LinearSegmentedColormap.from_list("", ["royalblue","goldenrod"])
    
    plt.scatter(df_ant['y_memAvg(MB)'],df_ant['n_cores'],c=df_ant['combination_id'], edgecolors='none', cmap=new_cool)
    cb = plt.colorbar(ticks=df_grouped.sort_values(['combination_id'])['combination_id'].tolist())
    cb.ax.set_yticklabels(df_grouped.sort_values(['combination_id'])['combination'].tolist()) 
    plt.title(f'{alg}')
    plt.xlabel('Memory')
    plt.ylabel('#CPU Cores')
    plt.savefig(f'{alg}_scatterplot.png')
    plt.show()
    datalist.append([df_ant['y_memAvg(MB)'], df_ant['n_cores'], df_ant['combination_id']])
    
fig = plt.figure(figsize=(12,6))
fig.dpi=200

new_cool = matplotlib.colors.LinearSegmentedColormap.from_list("", ["royalblue","goldenrod"])

plt.scatter(datalist[0][0],datalist[0][1],c=df_ant['combination_id'], s=120, marker='*', edgecolors='k', lw=0.2, cmap=new_cool, label="ANTICIPATE", alpha=0.7)
plt.scatter(datalist[1][0],datalist[1][1],c=df_ant['combination_id'], edgecolors='k', lw=0.4, cmap=new_cool, label="CONTINGENCY", alpha=0.7)
cb = plt.colorbar(ticks=df_grouped.sort_values(['combination_id'])['combination_id'].tolist())
cb.ax.set_yticklabels(df_grouped.sort_values(['combination_id'])['combination'].tolist()) 
#plt.title(f'{alg}')
plt.legend()
plt.xlabel('Memory')
plt.ylabel('#CPU Cores')
plt.savefig(f'general_scatterplot.png')
plt.show()
