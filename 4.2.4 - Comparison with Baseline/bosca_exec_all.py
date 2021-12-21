# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 10:37:45 2021

@author: Andrea
"""

import os
import sys
import cplex
import docplex
import pandas as pd
import numpy as np
import pickle
import regex
import csv
from IPython.display import display
from eml.backend import cplex_backend
from eml.tree.reader import sklearn_reader
from eml.tree import embed as tembed
import docplex.mp.model as cpx
import docplex.mp.model_reader as mr
import time, datetime
pd.set_option('display.max_columns', 15)

embedding_time = 98.56631708145142 - 0.07569432258605957 # derived from the script build_model
solve_times = []

algs = {}
algs['ANTICIPATE'] = {}
algs['CONTINGENCY'] = {}


df = pd.read_csv('../testBaselineGreedy.csv')
df.columns = ['InstanceID', 'sol(keuro)', 'time(sec)', 'memAvg(MB)', 'PV(kW)', 'Load(kW)']
df['InstanceID'] = df['InstanceID'].astype(int)
df['PV(kW)'] = df['PV(kW)'].map(lambda entry: entry[1:-1].split())
df['PV(kW)'] = df['PV(kW)'].map(lambda entry: list(np.float_(entry)))
df['Load(kW)'] = df['Load(kW)'].map(lambda entry: entry[1:-1].split())
df['Load(kW)'] = df['Load(kW)'].map(lambda entry: list(np.float_(entry)))

df['PV_mean'] = df['PV(kW)'].map(lambda entry: np.array(entry).mean())
df['PV_std'] = df['PV(kW)'].map(lambda entry: np.array(entry).std())
df['Load_mean'] = df['Load(kW)'].map(lambda entry: np.array(entry).mean())
df['Load_std'] = df['Load(kW)'].map(lambda entry: np.array(entry).std())

del df['PV(kW)'] 
del df['Load(kW)']


current_row = []
columns = ['InstanceID', 'improvement', 'y_sol(keuro)_baseline', 'y_memAvg(MB)_baseline', 'y_time(sec)_baseline',\
            'b_ANTICIPATE', 'b_CONTINGENCY',
            'y_ANTICIPATE_memAvg(MB)', 'y_ANTICIPATE_time(sec)', 'y_ANTICIPATE_sol(keuro)',\
            'y_CONTINGENCY_memAvg(MB)', 'y_CONTINGENCY_time(sec)', 'y_CONTINGENCY_sol(keuro)',\
            'y_nTraces', 'y_nScenarios',\
            'y_PV_mean', 'y_PV_std', 'y_Load_mean', 'y_Load_std',\
            'embedding_time', 'solve_time', '#constraints', '#variables']


std = eval(sys.argv[1])
perc = int(sys.argv[2])

modread = mr.ModelReader()

final_df = pd.DataFrame()

if std != -1 and perc != -1:
    dirname = f'partials_space_{std}_{perc}'
else:
    dirname = 'partials_space'

if not os.path.exists(dirname):
    os.mkdir(dirname)



iteration = 0
improvements = [0.02, 0.05, 0.07, 0.10, 0.12, 0.15, 0.17, 0.20, 0.22, 0.25, 0.27, 0.30]
baseline = 0
mdl = modread.read_model("vpp_model_time.sav")
for i in range(30):
    baseline = df['sol(keuro)'].iloc[i]

    for improvement in improvements:
        if os.path.isfile('./{}/general_model_{}_partial_improvement{}_instance{}.csv'.format(dirname, iteration, improvement, i)):
            iteration += 1
            continue  
        
        mdl.add_constraint(mdl.get_var_by_name('y_PV_mean') == df['PV_mean'].iloc[i], '1')
        mdl.add_constraint(mdl.get_var_by_name('y_PV_std') == df['PV_std'].iloc[i], '2')
        mdl.add_constraint(mdl.get_var_by_name('y_Load_mean') == df['Load_mean'].iloc[i], '3')
        mdl.add_constraint(mdl.get_var_by_name('y_Load_std') == df['Load_std'].iloc[i], '4')
        
        # Constraint da linearizzare
        # mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}') * mdl.get_var_by_name(f'b_{alg}') <= v)
        for alg in algs.keys():
            cost_coeff = 0
            
            if std != -1 and perc != -1:
                try:
                    coeffs = pd.read_csv(f'../{alg}_coefficients_{std}_{perc}_pruned.csv')
                    cost_coeff = coeffs['Cost'].iloc[0]*std*coeffs['Cost'].iloc[1]
                except:
                    pass
            
            #mdl.add_constraint(mdl.get_var_by_name('y_time(sec)') <= float(time_b))
            mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_sol(keuro)') <= baseline - (baseline*improvement) - cost_coeff, f'5_{alg}')
            mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_sol(keuro)').lb * mdl.get_var_by_name(f'b_{alg}') <= baseline - (baseline*improvement)- cost_coeff, f'6_{alg}')
            
        
        num_constr = mdl.number_of_constraints
        num_vars = mdl.number_of_variables
        
        solve_time = time.time()
        sol = mdl.solve()
        solve_time = time.time() - solve_time
        print(solve_time)
        print("Improvement {}, instance {} done".format(improvement, i))
        if sol is None:
            result_data = [i, improvement, baseline, df['memAvg(MB)'].iloc[i], df['time(sec)'].iloc[i]] + ['no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol'] +\
                [df['PV_mean'].iloc[i], df['PV_std'].iloc[i],\
                 df['Load_mean'].iloc[i], df['Load_std'].iloc[i], embedding_time, solve_time, num_constr, num_vars]
            df_tmp = pd.DataFrame(data=result_data).T
            df_tmp.columns = columns
            df_tmp.to_csv('./{}/general_model_{}_partial_improvement{}_instance{}.csv'.format(dirname,iteration, improvement, i), index=None)
        else:
            sol_df = sol.as_df().T
            sol_df.columns = sol_df.iloc[0].to_list()
            try:
                A_b = sol_df.iloc[1]['b_ANTICIPATE']
            except:
                A_b = 0
            try:
                C_b = sol_df.iloc[1]['b_CONTINGENCY']
            except:
                C_b = 0
            try:
                scenarios = sol_df.iloc[1]['y_nScenarios']
            except:
                scenarios = 0
            try:
                A_tim = sol_df.iloc[1]['y_ANTICIPATE_time(sec)']
            except:
                A_tim = 0
            try:
                A_sol = sol_df.iloc[1]['y_ANTICIPATE_sol(keuro)']
            except:
                A_sol = 0
            try:
                A_mem = sol_df.iloc[1]['y_ANTICIPATE_memAvg(MB)']
            except:
                A_mem = 0
            try:
                traces = sol_df.iloc[1]['y_nTraces']
            except:
                traces = 0
            try:
                C_tim = sol_df.iloc[1]['y_CONTINGENCY_time(sec)']
            except:
                C_tim = 0
            try:
                C_sol = sol_df.iloc[1]['y_CONTINGENCY_sol(keuro)']
            except:
                C_sol = 0
            try:
                C_mem = sol_df.iloc[1]['y_CONTINGENCY_memAvg(MB)']
            except:
                C_mem = 0
            result_data = [i, improvement, baseline, df['memAvg(MB)'].iloc[i], df['time(sec)'].iloc[i],\
                           A_b, C_b,\
                           A_mem, \
                           A_tim, \
                           A_sol,\
                           C_mem, \
                           C_tim, \
                           C_sol,\
                           traces,\
                           scenarios,\
                           sol_df.iloc[1]['y_PV_mean'],\
                           sol_df.iloc[1]['y_PV_std'],\
                           sol_df.iloc[1]['y_Load_mean'],\
                           sol_df.iloc[1]['y_Load_std'],\
                           embedding_time, solve_time,\
                           num_constr, num_vars]
            
            df_tmp = pd.DataFrame(data=result_data).T
            df_tmp.columns = columns
            
            df_tmp.to_csv('./{}/general_model_{}_partial_improvement{}_instance{}.csv'.format(dirname, iteration, improvement, i), index=None)
        iteration += 1
        
        mdl.remove_constraint('1')
        mdl.remove_constraint('2')
        mdl.remove_constraint('3')
        mdl.remove_constraint('4')
        for alg in algs.keys():
            mdl.remove_constraint(f'5_{alg}')
            mdl.remove_constraint(f'6_{alg}')
        
        
