# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 13:35:02 2021

@author: Andrea
"""

import pandas as pd
import matplotlib.pyplot as plt

files = ['baseline_improvement_general_model_prepr.csv', 'baseline_improvement_general_model_std1_perc90_filtered_errs_prepr.csv', 'baseline_improvement_general_model_std1_perc95_filtered_errs_prepr.csv']
titles = ['Objective: minimize time', 'Objective: minimize time, q = 90', 'Objective: minimize time, q = 95']
filenames = ['baseline_comparison_objminTime', 'baseline_comparison_objminTime_std_90-perc', 'baseline_comparison_objminTime_std_95-perc']
scale = 'log'
for file, title, fn in zip(files, titles, filenames):
    df = pd.read_csv(file)
    df['new_mem'] = df['y_memAvg(MB)'] / df['y_memAvg(MB)_baseline']
    df_proc = pd.DataFrame(data=[df['InstanceID'], df['improvement'], df['new_mem'], df['alg']]).T
    print(df_proc)
    
    instances_values = []
    ANT_markers = []
    CONT_markers = []
    for n in [2, 5, 15, 19, 28]:
        X = []
        Y = []
        Z = []
        ANT_x = []
        ANT_y = []
        CONT_x = []
        CONT_y = []
        for index, row in df_proc.iterrows():
            if row['InstanceID'] == n:
                X.append(row['improvement'])
                Y.append(row['new_mem'])
                Z.append(row['alg'])
                if row['alg'] == 'A':
                    ANT_x.append(row['improvement'])
                    ANT_y.append(row['new_mem'])
                else:
                    CONT_x.append(row['improvement'])
                    CONT_y.append(row['new_mem'])
        instances_values.append([X, Y, Z])
        ANT_markers.append([ANT_x, ANT_y])
        CONT_markers.append([CONT_x, CONT_y])
    
    colors = ['blue', 'orange', 'green', 'red', 'turquoise']
    # Plot lines with different marker sizes
    fig = plt.figure(figsize=(5, 5))
    for instance in range(5):
        plt.scatter(ANT_markers[instance][0], ANT_markers[instance][1], c=colors[instance], marker='P', edgecolors='k', linewidths=0.4, zorder=10)
        plt.scatter(CONT_markers[instance][0], CONT_markers[instance][1], c=colors[instance], marker='^', edgecolors='k', linewidths=0.4, zorder=10)
    plt.plot(instances_values[0][0], instances_values[0][1], label = '#2', color=colors[0], lw=1.2, ms=6, markeredgewidth=0.2, markeredgecolor='k') # 2
    plt.plot(instances_values[1][0], instances_values[1][1], label = '#5', color=colors[1], lw=1.2, ms=6, markeredgewidth=0.2, markeredgecolor='k') # 5
    plt.plot(instances_values[2][0], instances_values[2][1], label = '#15', color=colors[2], lw=1.2, ms=6, markeredgewidth=0.2, markeredgecolor='k') # 15
    plt.plot(instances_values[3][0], instances_values[3][1], label = '#19', color=colors[3], lw=1.2, ms=6, markeredgewidth=0.2, markeredgecolor='k') # 19
    plt.plot(instances_values[4][0], instances_values[4][1], label = '#28', color=colors[4], lw=1.2, ms=6, markeredgewidth=0.2, markeredgecolor='k') # 28
    
    
    plt.xlabel('Improvement', fontsize=14)
    plt.ylabel('Memory increase', fontsize=14)
    #plt.yscale('log')
    handles, labels = plt.gca().get_legend_handles_labels()
    import matplotlib.lines as mlines
    
    mark_ANT = mlines.Line2D([], [], color='white', marker='P', linestyle='None',
                              markersize=8, label='ANTICIPATE', markeredgewidth=0.4, markeredgecolor='k')
    mark_CONT = mlines.Line2D([], [], color='white', marker='^', linestyle='None',
                              markersize=8, label='CONTINGENCY', markeredgewidth=0.4, markeredgecolor='k')
    
    handles.extend([mark_ANT,mark_CONT])
    
    plt.legend(handles=handles, title=r'$\bf{Instance}$', fontsize = 'medium', title_fontsize='medium', prop={"weight":"bold"}, markerscale=1.2)
    #plt.title(title)
    if scale == 'log':
        plt.yscale('log')
        plt.show()
        fig.savefig(f'{fn}_mem_increase_logscale.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()
        fig.savefig(f'{fn}_mem_increase.png', dpi=300, bbox_inches='tight')
        
    
    
    df = pd.read_csv(file)
    df['new_time'] = df['y_time(sec)'] / df['y_time(sec)_baseline']
    df_proc = pd.DataFrame(data=[df['InstanceID'], df['improvement'], df['new_time'], df['alg']]).T
    print(df_proc)
    
    instances_values = []
    ANT_markers = []
    CONT_markers = []
    for n in [2, 5, 15, 19, 28]:
        X = []
        Y = []
        Z = []
        ANT_x = []
        ANT_y = []
        CONT_x = []
        CONT_y = []
        for index, row in df_proc.iterrows():
            if row['InstanceID'] == n:
                X.append(row['improvement'])
                Y.append(row['new_time'])
                Z.append(row['alg'])
                if row['alg'] == 'A':
                    ANT_x.append(row['improvement'])
                    ANT_y.append(row['new_time'])
                else:
                    CONT_x.append(row['improvement'])
                    CONT_y.append(row['new_time'])
        instances_values.append([X, Y, Z])
        ANT_markers.append([ANT_x, ANT_y])
        CONT_markers.append([CONT_x, CONT_y])
        
    colors = ['blue', 'orange', 'green', 'red', 'turquoise']
    # Plot lines with different marker sizes
    fig = plt.figure(figsize=(5, 5))
    for instance in range(5):
        plt.scatter(ANT_markers[instance][0], ANT_markers[instance][1], c=colors[instance], marker='P', edgecolors='k', linewidths=0.4, zorder=10)
        plt.scatter(CONT_markers[instance][0], CONT_markers[instance][1], c=colors[instance], marker='^', edgecolors='k', linewidths=0.4, zorder=10)
    plt.plot(instances_values[0][0], instances_values[0][1], label = '#2', color=colors[0], lw=1.2, ms=6, markeredgewidth=0.2, markeredgecolor='k') # 2
    plt.plot(instances_values[1][0], instances_values[1][1], label = '#5', color=colors[1], lw=1.2, ms=6, markeredgewidth=0.2, markeredgecolor='k') # 5
    plt.plot(instances_values[2][0], instances_values[2][1], label = '#15', color=colors[2], lw=1.2, ms=6, markeredgewidth=0.2, markeredgecolor='k') # 15
    plt.plot(instances_values[3][0], instances_values[3][1], label = '#19', color=colors[3], lw=1.2, ms=6, markeredgewidth=0.2, markeredgecolor='k') # 19
    plt.plot(instances_values[4][0], instances_values[4][1], label = '#28', color=colors[4], lw=1.2, ms=6, markeredgewidth=0.2, markeredgecolor='k') # 28
    
    plt.xlabel('Improvement', fontsize=14)
    plt.ylabel('Time increase', fontsize=14)
    #plt.yscale('log')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    import matplotlib.lines as mlines
    
    mark_ANT = mlines.Line2D([], [], color='white', marker='P', linestyle='None',
                              markersize=8, label='ANTICIPATE', markeredgewidth=0.4, markeredgecolor='k')
    mark_CONT = mlines.Line2D([], [], color='white', marker='^', linestyle='None',
                              markersize=8, label='CONTINGENCY', markeredgewidth=0.4, markeredgecolor='k')
    
    handles.extend([mark_ANT,mark_CONT])
    
    
    plt.legend(handles=handles, title=r'$\bf{Instance}$', fontsize = 'medium', title_fontsize='medium', prop={"weight":"bold"}, markerscale=1.2)
    #plt.title(title)
    
    if scale == 'log':
        plt.yscale('log')
        plt.show()
        fig.savefig(f'{fn}_time_increase_logscale.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()
        fig.savefig(f'{fn}_time_increase.png', dpi=300, bbox_inches='tight')