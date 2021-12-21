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

embedding_time = 105.626060009003 - 0.17548632621765137 # derived from time_logs.csv, (after_DT_time - before_DT_time) 
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
columns = ['time_less_than', \
            'b_ANTICIPATE', 'b_CONTINGENCY',
            'y_ANTICIPATE_memAvg(MB)', 'y_ANTICIPATE_time(sec)', 'y_ANTICIPATE_sol(keuro)',\
            'y_CONTINGENCY_memAvg(MB)', 'y_CONTINGENCY_time(sec)', 'y_CONTINGENCY_sol(keuro)',\
            'y_nTraces', 'y_nScenarios',\
            'y_PV_mean', 'y_PV_std', 'y_Load_mean', 'y_Load_std',\
            'embedding_time', 'solve_time', '#constraints', '#variables', 'ANTICIPATE_n_cores', 'CONTINGENCY_n_cores', 'y_ANTICIPATE_par_time', 'y_CONTINGENCY_par_time']

time_bounds = ['no', '1', '2', '3', '5', '8', '10', '20', '30', '60', '100', '120', '200', '300', '500']

modread = mr.ModelReader()

final_df = pd.DataFrame()

dirname = f'partials_space_{fixed_alg}'

if not os.path.exists(dirname):
    os.mkdir(dirname)
 

mdl = None

iteration = 0
for time_b in time_bounds:
    for i in range(30):
        if os.path.isfile('./{}/general_model_{}_partial_time({})_instance{}.csv'.format(dirname, iteration, time_b, i)):
            iteration += 1
            continue

        baseline = df['baseline_sol(keuro)'].iloc[i]
        
        # Init model just once
        if mdl == None:
            mdl = modread.read_model("vpp_model.sav")
        #mdl = modread.read_model("vpp_model_parallel_dt10.sav")
        
        mdl.add_constraint(mdl.get_var_by_name(f'b_{fixed_alg}') == 1, ctname='0') # impose usage of only one algorithm 
        mdl.add_constraint(mdl.get_var_by_name('y_PV_mean') == df['PV_mean'].iloc[i], ctname='1')
        mdl.add_constraint(mdl.get_var_by_name('y_PV_std') == df['PV_std'].iloc[i], ctname='2')
        mdl.add_constraint(mdl.get_var_by_name('y_Load_mean') == df['Load_mean'].iloc[i], ctname='3')
        mdl.add_constraint(mdl.get_var_by_name('y_Load_std') == df['Load_std'].iloc[i], ctname='4')
        
        # Constraints linearization
        # mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}') * mdl.get_var_by_name(f'b_{alg}') <= v)
        for alg in algs.keys():
            if time_b != 'no':
                  # no robust and for time only
                  mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_par_time') <= float(time_b), ctname=f'5_{alg}')
                  mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_par_time').lb * 
                      mdl.get_var_by_name(f'b_{alg}') <= float(time_b),ctname=f'6_{alg}')
        
        num_constr = mdl.number_of_constraints
        num_vars = mdl.number_of_variables
        
        solve_time = time.time()
        print("Combination time bound {}, instance {}; TRYING TO SOLVE...".format(time_b, i))
        sol = mdl.solve()
        if sol == None:
            print("Got None sol, trying again...")
            solve_time = time.time()
            sol = mdl.solve()
                
        solve_time = time.time() - solve_time
        print(solve_time)
        if sol != None:
            print("Combination time bound {}, instance {} done".format(time_b, i))
        else:
            print("Combination time bound {}, instance {}: got no solution".format(time_b, i))
        if sol is None:
            result_data = [time_b] + ['no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol', 'no_sol'] +\
                [df['PV_mean'].iloc[i], df['PV_std'].iloc[i],\
                 df['Load_mean'].iloc[i], df['Load_std'].iloc[i], embedding_time, solve_time, num_constr, num_vars, 'no_sol', 'no_sol', 'no_sol', 'no_sol']
            df_tmp = pd.DataFrame(data=result_data).T
            df_tmp.columns = columns
            df_tmp.to_csv('./{}/general_model_{}_partial_time({})_instance{}.csv'.format(dirname,iteration, time_b, i), index=None)
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
            try:
                A_num_cores = sol_df.iloc[1]['ANTICIPATE_n_cores']
            except:
                A_num_cores = 0
            try:
                C_num_cores = sol_df.iloc[1]['CONTINGENCY_n_cores']
            except:
                C_num_cores = 0
            try:
                A_par_time = sol_df.iloc[1]['y_ANTICIPATE_par_time']
            except:
                A_par_time = 0
            try:
                C_par_time = sol_df.iloc[1]['y_CONTINGENCY_par_time']
            except:
                C_par_time = 0
            result_data = [time_b, \
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
                           num_constr, num_vars, A_num_cores, C_num_cores, A_par_time, C_par_time]
            
            df_tmp = pd.DataFrame(data=result_data).T
            df_tmp.columns = columns
            
            df_tmp.to_csv('./{}/general_model_{}_partial_time({})_instance{}.csv'.format(dirname, iteration, time_b, i), index=None)
        iteration += 1
        
        mdl.remove_constraint('0')
        mdl.remove_constraint('1')
        mdl.remove_constraint('2')
        mdl.remove_constraint('3')
        mdl.remove_constraint('4')
        
        if time_b != 'no':
            for alg in algs.keys():
                mdl.remove_constraint(f'5_{alg}')
                mdl.remove_constraint(f'6_{alg}')
