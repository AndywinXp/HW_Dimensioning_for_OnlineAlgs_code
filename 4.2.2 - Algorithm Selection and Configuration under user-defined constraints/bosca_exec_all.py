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


df = pd.read_csv('InstancesTest.csv')
df.columns = ['InstanceID', 'PV(kW)', 'Load(kW)']
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
columns = ['time_less_than', 'mem_less_than', \
            'b_ANTICIPATE', 'b_CONTINGENCY',
            'y_ANTICIPATE_memAvg(MB)', 'y_ANTICIPATE_time(sec)', 'y_ANTICIPATE_sol(keuro)',\
            'y_CONTINGENCY_memAvg(MB)', 'y_CONTINGENCY_time(sec)', 'y_CONTINGENCY_sol(keuro)',\
            'y_nTraces', 'y_nScenarios',\
            'y_PV_mean', 'y_PV_std', 'y_Load_mean', 'y_Load_std',\
            'embedding_time', 'solve_time', '#constraints', '#variables']

bounds_space = sys.argv[1]

if bounds_space == 'contingency':
    time_bounds = ['no', '5', '30', '60']
    mem_bounds = ['no', '83', '87']
else:
    time_bounds = ['no', '30', '60', '120', '200', '300']
    mem_bounds = ['no', '100', '150', '200', '300']

std = int(sys.argv[2])
perc = int(sys.argv[3])

modread = mr.ModelReader()

final_df = pd.DataFrame()

if std != -1 and perc != -1:
    dirname = f'partials_space_{bounds_space}_{std}_{perc}'
else:
    dirname = f'partials_space_{bounds_space}'

if not os.path.exists(dirname):
    os.mkdir(dirname)



iteration = 0
for time_b in time_bounds:
    for mem_b in mem_bounds:   
        for i in range(30):
            if os.path.isfile('./{}/general_model_{}_partial_{}-{}_instance{}.csv'.format(dirname, iteration, time_b, mem_b, i)):
                iteration += 1
                continue
                
            mdl = modread.read_model("vpp_model.sav")
            
            mdl.add_constraint(mdl.get_var_by_name('y_PV_mean') == df['PV_mean'].iloc[i])
            mdl.add_constraint(mdl.get_var_by_name('y_PV_std') == df['PV_std'].iloc[i])
            mdl.add_constraint(mdl.get_var_by_name('y_Load_mean') == df['Load_mean'].iloc[i])
            mdl.add_constraint(mdl.get_var_by_name('y_Load_std') == df['Load_std'].iloc[i])
            
            # Constraint da linearizzare
            # mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}') * mdl.get_var_by_name(f'b_{alg}') <= v)
            for alg in algs.keys():
                time_coeff = 0
                mem_coeff = 0
                
                if std != -1 and perc != -1:
                    try:
                        coeffs = pd.read_csv(f'{alg}_coefficients_{std}_{perc}.csv')
                        time_coeff = coeffs['Time'].iloc[0]*coeffs['Time'].iloc[1]
                        mem_coeff = coeffs['Memory'].iloc[0]*coeffs['Memory'].iloc[1]
                    except:
                        pass
                
                if time_b != 'no':
                    #mdl.add_constraint(mdl.get_var_by_name('y_time(sec)') <= float(time_b))
                    mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_time(sec)') <= float(time_b) - time_coeff)
                    mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_time(sec)').lb * mdl.get_var_by_name(f'b_{alg}') <= float(time_b) - time_coeff)
                
                if mem_b != 'no':
                    #mdl.add_constraint(mdl.get_var_by_name('y_memAvg(MB)') <= float(mem_b))
                    mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_memAvg(MB)') <= float(mem_b) - mem_coeff)
                    mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_memAvg(MB)').lb * mdl.get_var_by_name(f'b_{alg}') <= float(mem_b) - mem_coeff)
            
            num_constr = mdl.number_of_constraints
            num_vars = mdl.number_of_variables
            
            solve_time = time.time()
            sol = mdl.solve()
            solve_time = time.time() - solve_time
            print(solve_time)
            print("Combination {}-{}, instance {} done".format(time_b, mem_b, i))
            if sol is None:
                result_data = [time_b, mem_b] + ['no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol'] +\
                    [df['PV_mean'].iloc[i], df['PV_std'].iloc[i],\
                     df['Load_mean'].iloc[i], df['Load_std'].iloc[i], embedding_time, solve_time, num_constr, num_vars]
                df_tmp = pd.DataFrame(data=result_data).T
                df_tmp.columns = columns
                df_tmp.to_csv('./{}/general_model_{}_partial_{}-{}_instance{}.csv'.format(dirname,iteration, time_b, mem_b, i), index=None)
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
                result_data = [time_b, mem_b, \
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
                
                df_tmp.to_csv('./{}/general_model_{}_partial_{}-{}_instance{}.csv'.format(dirname, iteration, time_b, mem_b, i), index=None)
            iteration += 1
            del mdl
