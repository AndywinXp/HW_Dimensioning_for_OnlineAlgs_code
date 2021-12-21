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

embedding_time = 98.595203676224 - 0.10444235801696777 # derived from time_logs.csv, (after_DT_time - before_DT_time) 
solve_times = []

if len(sys.argv) <= 1:
    print('No algorithm specified, quitting...')
    sys.exit()

fixed_alg = sys.argv[1]

if fixed_alg != 'ANTICIPATE' and fixed_alg != 'CONTINGENCY':
    print('Argument error, quitting...')
    sys.exit()

algs = {}
algs['ANTICIPATE'] = {}
algs['CONTINGENCY'] = {}


df = pd.read_csv('testBaselineGreedy.csv')
df.columns = ['InstanceID', 'baseline_sol(keuro)', 'baseline_time(sec)', 'baseline_memAvg(MB)', 'PV(kW)', 'Load(kW)']
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

time_bounds = ['no', '60', '120', '300']
sol_bounds = ['no', '0.05', '0.10', '0.20'] # improvements over baseline

std = 1
perc = 90

modread = mr.ModelReader()

final_df = pd.DataFrame()

if std != -1 and perc != -1:
    dirname = f'partials_space_{fixed_alg}_{std}_{perc}'
else:
    dirname = f'partials_space_{fixed_alg}'

if not os.path.exists(dirname):
    os.mkdir(dirname)
 
for alg in algs.keys():
    if std != -1 and perc != -1:
        try:
            # DT10 for time, DT15 for cost
            coeffs = pd.read_csv(f'{alg}_coefficients_{std}_{perc}.csv')
            time_coeff = coeffs['Time'].iloc[0]*coeffs['Time'].iloc[1]
            cost_coeff = coeffs['Cost'].iloc[0]*coeffs['Cost'].iloc[1]
        except:
            print(f'ERROR: Couldn\'t find coefficients file {alg}_coefficients_{std}_{perc}.csv, quitting...')
            sys.exit()

mdl = None

iteration = 0
for time_b in time_bounds:
    for sol_b in sol_bounds:   
        for i in range(30):
            if os.path.isfile('./{}/general_model_{}_partial_time{}-sol{}%_instance{}.csv'.format(dirname, iteration, time_b, sol_b, i)):
                iteration += 1
                continue
            baseline = df['baseline_sol(keuro)'].iloc[i]
            
            # Init model just once
            if mdl == None:
                mdl = modread.read_model("vpp_model.sav")
                mdl.add_constraint(mdl.get_var_by_name(f'b_{fixed_alg}') == 1) # impose usage of only one algorithm
            
            mdl.add_constraint(mdl.get_var_by_name('y_PV_mean') == df['PV_mean'].iloc[i], ctname='1')
            mdl.add_constraint(mdl.get_var_by_name('y_PV_std') == df['PV_std'].iloc[i], ctname='2')
            mdl.add_constraint(mdl.get_var_by_name('y_Load_mean') == df['Load_mean'].iloc[i], ctname='3')
            mdl.add_constraint(mdl.get_var_by_name('y_Load_std') == df['Load_std'].iloc[i], ctname='4')
            
            # Constraints linearization
            # mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}') * mdl.get_var_by_name(f'b_{alg}') <= v)
            for alg in algs.keys():
                time_coeff = 0
                mem_coeff = 0
                
                if std != -1 and perc != -1:
                    try:
                        # DT10 for time, DT15 for cost
                        coeffs = pd.read_csv(f'{alg}_coefficients_{std}_{perc}.csv')
                        time_coeff = coeffs['Time'].iloc[0]*coeffs['Time'].iloc[1]
                        cost_coeff = coeffs['Cost'].iloc[0]*coeffs['Cost'].iloc[1]
                    except:
                        print(f'ERROR: Couldn\'t find coefficients file {alg}_coefficients_{std}_{perc}.csv, quitting...')
                        sys.exit()
                
                if time_b != 'no':
                    #mdl.add_constraint(mdl.get_var_by_name('y_time(sec)') <= float(time_b))
                    mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_time(sec)') <= float(time_b) - time_coeff, ctname='5_{alg}')
                    mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_time(sec)').lb * mdl.get_var_by_name(f'b_{alg}') <= float(time_b) - time_coeff, ctname='6_{alg}')
                
                if sol_b != 'no':
                    #mdl.add_constraint(mdl.get_var_by_name('y_memAvg(MB)') <= float(mem_b))
                    mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_sol(keuro)') <= baseline - (baseline*float(sol_b)) - cost_coeff, ctname='7_{alg}')
                    mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_sol(keuro)').lb * mdl.get_var_by_name(f'b_{alg}') <= baseline - (baseline*float(sol_b)) - cost_coeff, ctname='8_{alg}')
            
            num_constr = mdl.number_of_constraints
            num_vars = mdl.number_of_variables
            
            solve_time = time.time()
            sol = mdl.solve()
            solve_time = time.time() - solve_time
            print(solve_time)
            print("Combination time{}-sol%{}, instance {} done".format(time_b, sol_b, i))
            if sol is None:
                result_data = [time_b, sol_b] + ['no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol'] +\
                    [df['PV_mean'].iloc[i], df['PV_std'].iloc[i],\
                     df['Load_mean'].iloc[i], df['Load_std'].iloc[i], embedding_time, solve_time, num_constr, num_vars]
                df_tmp = pd.DataFrame(data=result_data).T
                df_tmp.columns = columns
                df_tmp.to_csv('./{}/general_model_{}_partial_time{}-sol{}%_instance{}.csv'.format(dirname,iteration, time_b, sol_b, i), index=None)
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
                result_data = [time_b, sol_b, \
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
                
                df_tmp.to_csv('./{}/general_model_{}_partial_time{}-sol{}%_instance{}.csv'.format(dirname, iteration, time_b, sol_b, i), index=None)
            iteration += 1
            
            mdl.remove_constraint('1')
            mdl.remove_constraint('2')
            mdl.remove_constraint('3')
            mdl.remove_constraint('4')
            
            for alg in algs.keys():
                mdl.remove_constraint(f'5_{alg}')
                mdl.remove_constraint(f'6_{alg}')
                mdl.remove_constraint(f'7_{alg}')
                mdl.remove_constraint(f'8_{alg}')
