# -*- coding: utf-8 -*-
import os
import cplex
import docplex
import pandas as pd
import numpy as np
import pickle
import regex
import csv
import sys
from IPython.display import display
import docplex.mp.model as cpx
import time, datetime
import docplex.mp.model_reader as mr    
from eml.backend import cplex_backend
from eml.tree.reader import sklearn_reader
from eml.tree import embed as tembed

min_cores_ = 1
max_cores_ = 32
# map the number of cores to the corresponding index in the time_redux_ list
time_redux_map_ = {1: 0, 2: 1, 4: 2, 8: 3, 12: 4, 16: 5, 32: 6}
time_redux_ = [1, 1/1.35, 1/(41.6/24.2), 1/(245/100), 1/2.81, 1/3.1, 1/3.88]
cores_allowed_values_ = [1, 2, 4, 8, 12, 16, 32]

def get_time_redux_given_ncores(ncores):
    if ncores not in cores_allowed_values_:
        return None
    if ncores not in time_redux_map_.keys():
        return None
    return time_redux_[time_redux_map_[ncores]]

def print_log(string, log_file='log_shell.txt'):
  print(string)
  with open(log_file, mode='a+') as f:
    f.write(string + '\n') 


def write_logs(EML_times, sol, mdl, DT_vars, user_constraints, objective_type,
        objective_var, trees, model_name, log_path="/content/EML_results"):
  #### Log TIMES
  EML_times['after_solve_time'] = time.time()
  times = [EML_times['after_solve_time'] - EML_times['before_modelEM_time']]
  labels = ['tot_solve_time']
  for time_label in EML_times:
    labels.append(time_label)
    times.append(f"{EML_times[time_label] - EML_times['before_modelEM_time']}")
  times.append(mdl.solve_details.time)
  labels.append('CPLEX_time(sol)')
  with open(f"{log_path}/time_logs.csv", mode='a+') as results_file:
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', 
            quoting=csv.QUOTE_MINIMAL)
    results_writer.writerow(labels)
    results_writer.writerow(times)

  #### Log DATA & SOLUTION
  # data
  trees_str = list(trees.keys())
  objective_str = f"{objective_type}({objective_var})"
  user_constraints_strs = []
  for i in range(len(user_constraints['variable'])):
    c = user_constraints['variable'][i]
    t = user_constraints['type'][i]
    v = user_constraints['value'][i]
    user_constraints_strs.append(f'{c}{t}{v}')
  sols = [objective_str, user_constraints_strs, trees_str, model_name]
  labels = ['objective', 'contraints', 'trees', 'model_file']

  # solution
  if sol is None:
    sols = sols + ['No sol found']
    labels = labels + ['status']
  else:
    sols = sols + [f"{sol.solve_details.status}", '{:.2f}'.format(
        mdl.solve_details.time)]
    labels = labels + ['status', 'time']
    for var in DT_vars:
      labels.append(var)
      sols.append(f"{sol[var]}") 

  with open(f"{log_path}/sol_logs.csv", mode='a+') as results_file:
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', 
            quoting=csv.QUOTE_MINIMAL)
    results_writer.writerow(labels)
    results_writer.writerow(sols)


def build_and_solve_EML(df_mins, df_maxs, user_constraints,
        objective_type, objective_var, trees, trees_features_target,
        model_name=None, partial_model_name='partial', save_path='.',
        enable_var_type=False):
  f = open('vars_constr_num.txt', 'w')

  if not model_name:
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
    model_name = f"EML_model_{dt_string}"
  print_log(f'------------------------------------------------------------\
          - MODEL {model_name} -------------------------------------------\
          --------------------')

  '''Section: Build & solve CPLEX Model'''
  print_log('\n=== Building basic model')
  EML_times = {}
  EML_times['before_modelEM_time'] = time.time()

  # Build a backend object
  bkd = cplex_backend.CplexBackend()
  # Build a docplex model
  mdl = cpx.Model()
  f.write('MARKER 0:model_creation:{}:{}\n'.format(mdl.number_of_constraints, 
      mdl.number_of_variables))
  ###### Define problem variables #####
  # DT variables
  DT_vars = []
  DT_vars_int = []

  #DT_vars_names = common_cols
  DT_vars_names_in = inst_descr
  DT_vars_names_out = ml_trgt
  DT_vars_names_params = []

  DT_vars_int_names = []
  binary_vars_names = []
  
  for alg in algs.keys():
      for var in algs[alg]['alg_params'].values():
        DT_vars_names_params.append(var['name'])
        if var['type'] == int:
            DT_vars_int_names.append(var['name'])

  bin_list = []
  # Binary vars
  for alg in algs.keys():
      bin_list.append(mdl.binary_var(f"b_{alg}"))
      binary_vars_names.append(f"b_{alg}")

  mdl.add_constraint(mdl.sum(bin_list) == 1)
  
 
  ##### Load, convert & embed the DT #####
  model_path = f'{save_path}/'
  #### DT variables
  print_log("DTs variables")

  print_log("\tIndexed via alg")
  # insert variables that need to be indexed via the algorithm (e.g., the
  # targets of the ML models)
  for var in DT_vars_names_out:
      print_log(f"\t* {var}")
      for alg in algs.keys():
          print_log(f"\t\t* {alg}")
          DT_vars.append(mdl.continuous_var(lb = df_mins[alg].loc[var], 
              ub = df_maxs[alg].loc[var], name = f"y_{alg}_{var}"))

          print_log(f"\t\t * lb = {df_mins[alg].loc[var]}")
          print_log(f"\t\t * ub = {df_maxs[alg].loc[var]}")

  print_log("\tInstance description")
  # insert variables corresponing to instance descriptions
  for var in DT_vars_names_in:
      print_log(f"\t* {var}")
      DT_vars.append(mdl.continuous_var(lb = df_mins['glob'].loc[var], 
          ub = df_maxs['glob'].loc[var], name = f"y_{var}"))

  print_log("\tHparams")
  # insert variables corresponing to algoritms' hyperparameters
  for var in DT_vars_names_params:
      print_log(f"\t* {var}")
      DT_vars.append(mdl.continuous_var(lb = df_mins['glob'].loc[var], 
          ub = df_maxs['glob'].loc[var], name = f"y_{var}"))

      if var in DT_vars_int_names and enable_var_type:
          DT_vars_int.append(mdl.integer_var(lb = df_mins['glob'].loc[var], 
              ub = df_maxs['glob'].loc[var], name = f"y_{var}_int"))
          mdl.add_constraint(mdl.get_var_by_name(f'y_{var}_int') ==
                  mdl.get_var_by_name(f'y_{var}'))

  f.write('MARKER 2:after_DT_vars:{}:{}\n'.format(mdl.number_of_constraints, 
      mdl.number_of_variables))

  # Number of cores
  parallel_times_vars = {}
  n_cores = {}
  for alg in algs.keys():
      parallel_times_vars[alg] = []
      n_cores[alg] = mdl.integer_var(lb = min_cores_, ub = max_cores_, 
          name = f"{alg}_n_cores")
      print_log(f"\t\t* {alg}")
      parallel_times_vars[alg].append(mdl.continuous_var(
          #lb = 0,
          lb = df_mins['glob'].loc["time(sec)"]*time_redux_[-1],
          ub = df_maxs['glob'].loc["time(sec)"], name = f"y_{alg}_par_time"))
  
  # add constraints to force n_cores to assume only realistic values
  allowedVal_vars = {}
  for alg in algs.keys():
      allowedVal_vars[alg] = []
      for i in range(len(cores_allowed_values_)):
          allowedVal_vars[alg].append(mdl.binary_var(name=f"{alg}_av_{i}"))
      # Only one value is allowed for each variable
      bkd.cst_eq(mdl, bkd.xpr_sum(mdl, allowedVal_vars[alg]), 1)
      # assign the number of cores
      bkd.cst_eq(mdl, n_cores[alg], bkd.xpr_scalprod(
          mdl, cores_allowed_values_, allowedVal_vars[alg]))

  # add constraints to connect number of cores to runtime 
  # the y_{alg}_time is the runtime measured in the benchmarks and predicted by
  # the ML models -- all benchmarks used 1 core; y_{alg}_par_time is the
  # estimated runtime with different degrees of parallelism (estimated through
  # GUROBI official documentation)
  for alg in algs.keys():
      # this constraint is quadratic:
      #mdl.add_constraints([mdl.get_var_by_name(f'y_{alg}_par_time') == 
      #    mdl.get_var_by_name(f'y_{alg}_time(sec)') * time_redux_[i] *
      #    allowedVal_vars[alg][i] for i in range(len(time_redux_))])

      # we use indicator constraints instead
      mdl.add_indicator_constraints([
              mdl.indicator_constraint(allowedVal_vars[alg][i],
                  mdl.get_var_by_name(f'y_{alg}_par_time') == 
                  mdl.get_var_by_name(f'y_{alg}_time(sec)') * time_redux_[i])
                  for i in range(len(time_redux_))])

  ## Load DT
  print_log('\n=== Embedding DTs')
  EML_times['before_DT_time'] = time.time()
      
  # convert sklearn DT in EML format
  regr_em = {}
  for tree_name in trees.keys():
      regr_em[tree_name] = sklearn_reader.read_sklearn_tree(trees[tree_name])
      print_log(f"Tree {tree_name}. "
              f"Attributes: {regr_em[tree_name].attributes()} loaded. "
              f"Features {trees_features_target[tree_name]['features']}, "
              f"target {trees_features_target[tree_name]['target']}")
      
  # Handle bounds: bounds for the DT attributes need to be manually specified
  feature_names = ['nTraces', 'PV_mean', 'PV_std', 'Load_mean', 'Load_std']
  for tree_name in trees.keys(): 
      for attr in  regr_em[tree_name].attributes():
          regr_em[tree_name].update_ub(attr, 
                  df_maxs['glob'].loc[feature_names[attr]])
          regr_em[tree_name].update_lb(attr, 
                  df_mins['glob'].loc[feature_names[attr]])

  # Embed DT
  print_log('\nEmbedding trees...')
  for tree_name in trees.keys(): 
      print_log(f"\t* {tree_name}")
      tree_target_name = trees_features_target[tree_name]['target']
      #tree_target = DT_vars[DT_vars_names.index(tree_target_name)]
      #tree_target = DT_vars[DT_vars_names_out.index(tree_target_name)]
      alg = trees_features_target[tree_name]['alg']
      trgt = trees_features_target[tree_name]['target']
      tree_target = mdl.get_var_by_name(f'y_{alg}_{trgt}')
      tree_features = []
      for feat in trees_features_target[tree_name]['features']:
          print_log(f"\t\t* {feat}")
          #tree_features.append(DT_vars[DT_vars_names.index(feat)])
          if feat in DT_vars_names_in:
              tree_features.append(mdl.get_var_by_name(f'y_{feat}'))
          elif feat in DT_vars_names_params:
              tree_features.append(mdl.get_var_by_name(f'y_{feat}'))
          elif feat in DT_vars_names_out:
              alg = tree_name.split('_')[0]
              tree_features.append(mdl.get_var_by_name(f'y_{alg}_{feat}'))
          else:
              print_log(f"Problem! Features not found {feat}")
              sys.exit()

      tembed.encode_backward_implications(bkd, regr_em[tree_name], mdl, 
              tree_features,  tree_target, f'regr_DT_{tree_name}')

      f.write('MARKER 3:after_{}_DT_embedding:{}:{}\n'.format(tree_name, 
          mdl.number_of_constraints, mdl.number_of_variables))
      print_log(f'* Tree {tree_name} embedded')
      print_log(f"*       target: {tree_target}")
      print_log(f"*       features:{tree_features}")

  EML_times['after_DT_time'] = time.time()
      
  # Save partial model
  mdl.export_as_sav(model_path)
  f.close()
  print_log(f"\nPartial model saved to: {model_path}")

  ##### Define problem constraints & objective #####
  print_log('\n=== Adding custom constraints & objective')
  
  # Custom constraints
  print_log("Custom constraints:")
  for i in range(len(user_constraints['variable'])):
    c = user_constraints['variable'][i]
    t = user_constraints['type'][i]
    v = user_constraints['value'][i]
    print_log(f'\t* {c} {t} {v}')

    # we linearize the quadratic constraints (see commented first formulation)
    # using the suggestion found here:
    # http://yetanothermathprogrammingconsultant.blogspot.com/2008/05/multiplication-of-continuous-and-binary.html.
    # We assume that x_lo and x_up (y_{alg}_{c}.lb and y_{alg}_{c}.ub) are
    # greater than zero
    for alg in algs.keys():
        print_log(f"\t* {alg}")

        # suppose to have a set of coefficients for each algorithms
        # coeff = {}
        # coeff[c] = mem_coeff[alg]
        # coeff[c] = time_coeff[alg]
        # coeff[c] = cost_coeff[alg]

        if t == '<=':
          #mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}') * 
          #    mdl.get_var_by_name(f'b_{alg}') <= v)

          #mdl.add_constraint(-mdl.get_var_by_name(f'y_{alg}_{c}').ub * 
          #    mdl.get_var_by_name(f'b_{alg}') >= -v)
          #print(f"\t\t{alg}-{c} UB: {mdl.get_var_by_name(f'y_{alg}_{c}').ub}")
          #print(f"\t\t{alg}-{c} LB: {mdl.get_var_by_name(f'y_{alg}_{c}').lb}")
          #print(f"\t\t{v}")

          # no robust
          #mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}') <= v)
          #mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}').lb * 
          #    mdl.get_var_by_name(f'b_{alg}') <= v)

          # no robust and for time only
          mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_par_time') <= v)
          mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_par_time').lb * 
              mdl.get_var_by_name(f'b_{alg}') <= v)

          # robust 
          #mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}') <= v - coeff[c])
          #mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}').lb * 
          #    mdl.get_var_by_name(f'b_{alg}') <= v - coeff[c])
 
        elif t == '>=':
          #mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}') * 
          #    mdl.get_var_by_name(f'b_{alg}') >= v)

          # no robust
          mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}') >= v)
          mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}').ub * 
              mdl.get_var_by_name(f'b_{alg}') >= v)

          # robust
          #mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}') >= v + coeff[c])
          #mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}').ub * 
          #    mdl.get_var_by_name(f'b_{alg}') >= v + coeff[c])

        elif t == '==':
          #mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}') * 
          #    mdl.get_var_by_name(f'b_{alg}') == v)
          # no robust
          mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}').lb *
              mdl.get_var_by_name(f'b_{alg}') <= v)
          mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}').ub *
              mdl.get_var_by_name(f'b_{alg}') >= v)
          mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}') - 
                  mdl.get_var_by_name(f'y_{alg}_{c}').ub *
                  (1 - mdl.get_var_by_name(f'b_{alg}')) <= v)
          mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}') - 
                  mdl.get_var_by_name(f'y_{alg}_{c}').lb *
                  (1 - mdl.get_var_by_name(f'b_{alg}')) >= v)

          # robust
          #mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}').lb *
          #    mdl.get_var_by_name(f'b_{alg}') <= v - coeff[c])
          #mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}').ub *
          #    mdl.get_var_by_name(f'b_{alg}') >= v + coeff[c])
          #mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}') - 
          #        mdl.get_var_by_name(f'y_{alg}_{c}').ub *
          #        (1 - mdl.get_var_by_name(f'b_{alg}')) <= v - coeff[c])
          #mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{c}') - 
          #        mdl.get_var_by_name(f'y_{alg}_{c}').lb *
          #        (1 - mdl.get_var_by_name(f'b_{alg}')) >= v + coeff[c])
 
        else:
          print('Unsupported constraint type, terminating..')
          sys.exit()

  # Objective
  print_log("Objective:")

  # build a list of expressions y_{alg}_sol(keuro) * b_{alg} + {alg}_n_cores * b_{alg}
  prod_list = []

  prod_list_sol = []
  for alg in algs.keys():
      prod_list_sol.append((mdl.get_var_by_name(f'y_{alg}_sol(keuro)') * mdl.get_var_by_name(f'b_{alg}')))

  prod_list_core = []
  for alg in algs.keys():
      prod_list_core.append((mdl.get_var_by_name(f"{alg}_n_cores") * mdl.get_var_by_name(f'b_{alg}')))
  
  prod_list = prod_list_sol + prod_list_core
 
  print_log(f'\t* {objective_type}')
  if objective_type == 'min':
    mdl.minimize(mdl.sum(prod_list))
  else:
    mdl.maximize(mdl.sum(prod_list))

  EML_times['after_settings_time'] = time.time()
  
  
  with open('obtained_model_test.txt', "w") as output_file:
      output_file.write(mdl.export_as_lp_string())
  
  ##### Print info & save EML model #####
  print_log('\n=== Print info & save')
  vars = mdl.find_re_matching_vars(regex.compile(r'.*'))
  print_log(f'After embedding:\n* {len(vars)} VARIABLES\n\tbefore DT: ')
  display(mdl.find_re_matching_vars(regex.compile(r'^((?!DT).)*$')))
  #print_log(f'After embedding:\n* {len(vars)} VARIABLES\n\t: ')
  #display(mdl.find_re_matching_vars(regex.compile(r'^((?!DT).)*$')))
  ntot = 0
  for i, obj in enumerate(mdl.generate_user_linear_constraints()):
    ntot = ntot + 1
  print_log(f'* {ntot} LINEAR CONSTRAINTS\n before DT:')
  for i, obj in enumerate(mdl.generate_user_linear_constraints()):
    string = f"{obj}"
    if "DT" not in string:
      print_log(string)

  EML_times['after_print_time'] = time.time()

  # save model to be reused
  model_path = f'{save_path}/{model_name}'
  mdl.export_as_sav(model_path)
  print_log(f"\nModel saved to: {model_path}")

  EML_times['after_save_time'] = time.time()

  ## Print time data
  EML_times['after_modelEM_time'] = time.time()
  tot_time = EML_times['after_modelEM_time'] - EML_times['before_modelEM_time']
  print_log(f"\nTotal time needed to create MP model {tot_time}")
  for time_label in EML_times:
    t_time = EML_times[time_label] - EML_times['before_modelEM_time']
    print_log(f"* {time_label}: {t_time}")

  print_log('\n=== Model info')
  mdl.print_information()
  print_log('\n=== Model lp_string')
  mdl.lp_string
  mdl.export_as_lp(basename="my_model", path=".")

  ################################ Solve ################################
  print_log('\n=== Starting the solution process')
  mdl.set_time_limit(20000)
  sol = mdl.solve()

  # Print solution
  if sol is None:
      print_log('No solution found')
  else:
      print_log('SOLUTION DATA')
      print_log('Solution time: {:.2f} (sec)'.format(mdl.solve_details.time))
      print_log('Solver status: {}'.format(sol.solve_details.status))

      for var in DT_vars:
        print_log(f'Variable {var}: {sol[var]}')
      for alg in algs.keys():
        var = f'y_{alg}_par_time'
        print_log(f'Variable {var}: {sol[var]}')
      for var in DT_vars_int:
        print_log(f'Int variable {var}: {sol[var]}')
      for var in binary_vars_names:
        print_log(f'Binary variable {var}: {sol[var]}')
      for alg in algs.keys():
        for i in range(len(cores_allowed_values_)):
          var = f'{alg}_av_{i}'
          print_log(f'Aux. var. {var}: {sol[var]}')
        var = f'{alg}_n_cores'
        print_log(f'N.Cores {var}: {sol[var]}')

  # Log
  DT_vars = DT_vars + DT_vars_int
  write_logs(EML_times, sol, mdl, DT_vars, user_constraints, objective_type, 
          objective_var, trees, model_name, log_path=save_path)
  print_log(f'---------------------------------------------------------------\
          -------------------------------------------------------------------\
          ------------------')

# Common columns
#common_cols = ['memAvg(MB)', 'time(sec)', 'sol(keuro)', 'PV_mean', 'PV_std', 
#'Load_mean', 'Load_std']
ml_trgt = ['memAvg(MB)', 'time(sec)', 'sol(keuro)'] 
inst_descr = ['PV_mean', 'PV_std', 'Load_mean', 'Load_std']
common_cols = ml_trgt + inst_descr

# Algorithms
algs = {}

algs['ANTICIPATE'] = {}
algs['ANTICIPATE']['dataset_cols'] = ['nScenarios']
algs['ANTICIPATE']['alg_params'] = {'0': {'name': 'nScenarios', 'type': int}}

algs['ANTICIPATE']['trees'] = {}
algs['ANTICIPATE']['trees']['ANTICIPATE_input-cost_DecisionTree_MaxDepth15'] = {}
algs['ANTICIPATE']['trees']['ANTICIPATE_input-cost_DecisionTree_MaxDepth15']['features'
        ] = ['nScenarios', 'PV_mean', 'PV_std', 'Load_mean', 'Load_std']
algs['ANTICIPATE']['trees']['ANTICIPATE_input-cost_DecisionTree_MaxDepth15'
        ]['target'] = 'sol(keuro)'
algs['ANTICIPATE']['trees']['ANTICIPATE_input-cost_DecisionTree_MaxDepth15'
        ]['alg'] = 'ANTICIPATE'
algs['ANTICIPATE']['trees']['ANTICIPATE_no_input-memory_DecisionTree_MaxDepth10'] = {}
algs['ANTICIPATE']['trees']['ANTICIPATE_no_input-memory_DecisionTree_MaxDepth10'
        ]['features'] = ['nScenarios']
algs['ANTICIPATE']['trees']['ANTICIPATE_no_input-memory_DecisionTree_MaxDepth10'
        ]['target'] = 'memAvg(MB)'
algs['ANTICIPATE']['trees']['ANTICIPATE_no_input-memory_DecisionTree_MaxDepth10'
        ]['alg'] = 'ANTICIPATE'
algs['ANTICIPATE']['trees']['ANTICIPATE_no_input-time_DecisionTree_MaxDepth10'] = {}
algs['ANTICIPATE']['trees']['ANTICIPATE_no_input-time_DecisionTree_MaxDepth10'
        ]['features'] = ['nScenarios']
algs['ANTICIPATE']['trees']['ANTICIPATE_no_input-time_DecisionTree_MaxDepth10'
        ]['target'] = 'time(sec)'
algs['ANTICIPATE']['trees']['ANTICIPATE_no_input-time_DecisionTree_MaxDepth10'
        ]['alg'] = 'ANTICIPATE'

algs['CONTINGENCY'] = {}
algs['CONTINGENCY']['dataset_cols'] = ['nTraces']
algs['CONTINGENCY']['alg_params'] = {'0': {'name': 'nTraces', 'type': int}}

algs['CONTINGENCY']['trees'] = {}
algs['CONTINGENCY']['trees']['CONTINGENCY_input-cost_DecisionTree_MaxDepth15'] = {}
algs['CONTINGENCY']['trees']['CONTINGENCY_input-cost_DecisionTree_MaxDepth15']['features'
        ] = ['nTraces', 'PV_mean', 'PV_std', 'Load_mean', 'Load_std']
algs['CONTINGENCY']['trees']['CONTINGENCY_input-cost_DecisionTree_MaxDepth15'
        ]['target'] = 'sol(keuro)'
algs['CONTINGENCY']['trees']['CONTINGENCY_input-cost_DecisionTree_MaxDepth15'
        ]['alg'] = 'CONTINGENCY'
algs['CONTINGENCY']['trees']['CONTINGENCY_no_input-memory_DecisionTree_MaxDepth10'] = {}
algs['CONTINGENCY']['trees']['CONTINGENCY_no_input-memory_DecisionTree_MaxDepth10'
        ]['features'] = ['nTraces']
algs['CONTINGENCY']['trees']['CONTINGENCY_no_input-memory_DecisionTree_MaxDepth10'
        ]['target'] = 'memAvg(MB)'
algs['CONTINGENCY']['trees']['CONTINGENCY_no_input-memory_DecisionTree_MaxDepth10'
        ]['alg'] = 'CONTINGENCY'
algs['CONTINGENCY']['trees']['CONTINGENCY_no_input-time_DecisionTree_MaxDepth10'] = {}
algs['CONTINGENCY']['trees']['CONTINGENCY_no_input-time_DecisionTree_MaxDepth10'
        ]['features'] = ['nTraces']
algs['CONTINGENCY']['trees']['CONTINGENCY_no_input-time_DecisionTree_MaxDepth10'
        ]['target'] = 'time(sec)'
algs['CONTINGENCY']['trees']['CONTINGENCY_no_input-time_DecisionTree_MaxDepth10'
        ]['alg'] = 'CONTINGENCY'

globminlist = []
globmaxlist = []
globmaxdict = {}
globmindict = {}

## Variable bounds
for alg in algs.keys():
    """#### Load dataset"""
    df = pd.read_csv('{}_trainDataset.csv'.format(alg))
    
    # Removes header entries
    df = df[df['sol(keuro)'] != 'sol(keuro)']
    
    # Algorithm specific vars
    for var in algs[alg]['alg_params'].values():
        df[var['name']] = df[var['name']].astype(var['type'])
    
    # Fixed stuff which is always there
    df['PV(kW)'] = df['PV(kW)'].map(lambda entry: entry[1:-1].split())
    df['PV(kW)'] = df['PV(kW)'].map(lambda entry: list(np.float_(entry)))
    df['Load(kW)'] = df['Load(kW)'].map(lambda entry: entry[1:-1].split())
    df['Load(kW)'] = df['Load(kW)'].map(lambda entry: list(np.float_(entry)))
    df['sol(keuro)'] = df['sol(keuro)'].astype(float)
    df['time(sec)'] = df['time(sec)'].astype(float)
    df['memAvg(MB)'] = df['memAvg(MB)'].astype(float)
    
    df['PV_mean'] = df['PV(kW)'].map(lambda entry: np.array(entry).mean())
    df['PV_std'] = df['PV(kW)'].map(lambda entry: np.array(entry).std())
    df['Load_mean'] = df['Load(kW)'].map(lambda entry: np.array(entry).mean())
    df['Load_std'] = df['Load(kW)'].map(lambda entry: np.array(entry).std())
    
    cur_cols = algs[alg]['dataset_cols']
    cur_cols.extend(common_cols)
    df_s = df[cur_cols]
    
    globminlist.append(df_s.min())
    globmaxlist.append(df_s.max())

    globmaxdict[alg] = df_s.max()
    globmindict[alg] = df_s.min()

globmax = pd.DataFrame(globmaxlist).max()
globmin = pd.DataFrame(globminlist).min()

globmaxdict['glob'] = globmax
globmindict['glob'] = globmin

"""# Decision Trees

#### Load trees from the file system
"""

trees_files = {}
for alg in algs.keys():
    for tree_path in algs[alg]['trees'].keys():
        with open(tree_path, "rb") as output_file:
            print(f"Uploading tree {tree_path.split('/')[-1]}...")
            trees_files[tree_path.split('/')[-1]] = pickle.load(output_file)

trees_features_target = {}
for alg in algs.keys():
    trees_features_target = {**trees_features_target , **algs[alg]['trees']}

## Constraints
user_constraints = {}
user_constraints['variable'] = []#['time(sec)']
user_constraints['type'] =  []#'<=']
#user_constraints['value'] = [80, 135]
#user_constraints['value'] = [70, 90] # min functioning values
#user_constraints['value'] = [87] # min functioning values
user_constraints['value'] = []#[30] # min functioning values

## Objective Function
objective_var = 'sol(keuro)'
objective_type = 'min'

build_and_solve_EML(globmindict, globmaxdict, user_constraints, objective_type,
        objective_var, trees_files, trees_features_target,
        model_name="vpp_model", enable_var_type=True)
