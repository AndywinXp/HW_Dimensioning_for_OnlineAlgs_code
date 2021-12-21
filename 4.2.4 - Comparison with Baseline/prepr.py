# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 17:29:42 2021

@author: Andrea
"""

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 25)

files = ['baseline_improvement_general_model', 'baseline_improvement_general_model_std1_perc90_filtered_errs', 'baseline_improvement_general_model_std1_perc95_filtered_errs']
for file in files:
    df = pd.read_csv(f'{file}.csv')
    
    del df['#constraints']
    del df['#variables']
    del df['embedding_time']
    del df['solve_time']
    df = df[df['b_ANTICIPATE'] != 'no_sol']
    
    def getalg(row):
        if eval(row['b_ANTICIPATE']) == 1:
            return 'A'
        else:
            return 'C'
        
    def getsol(row):
        print(row['b_ANTICIPATE'])
        if eval(row['b_ANTICIPATE']) == 1:
            return row['y_ANTICIPATE_sol(keuro)']
        else:
            return row['y_CONTINGENCY_sol(keuro)']
        
    def getmem(row):
        print(row['b_ANTICIPATE'])
        if eval(row['b_ANTICIPATE']) == 1:
            return row['y_ANTICIPATE_memAvg(MB)']
        else:
            return row['y_CONTINGENCY_memAvg(MB)']
        
    def gettim(row):
        print(row['b_ANTICIPATE'])
        if eval(row['b_ANTICIPATE']) == 1:
            return row['y_ANTICIPATE_time(sec)']
        else:
            return row['y_CONTINGENCY_time(sec)']
        
    df['y_sol(keuro)'] = df.apply(lambda row : getsol(row), axis=1)
    df['y_memAvg(MB)'] = df.apply(lambda row : getmem(row), axis=1) 
    df['y_time(sec)'] = df.apply(lambda row : gettim(row), axis=1)
    df['alg'] = df.apply(lambda row : getalg(row), axis=1)
    print(df)
    
    df.to_csv(f'{file}_prepr.csv', index=None)