#!/usr/bin/env python
# coding: utf-8

import time
import sklearn
import pandas as pd
import numpy as np
import pickle
# plotting
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
# ML
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV # grid search for hyperparms tuning: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import sklearn.metrics
pd.set_option('display.max_columns', 15)


# #### Data prep

df_old = pd.read_csv('./trainDataset_old.csv')
df_new = pd.read_csv('./trainDataset.csv')
df = pd.concat([df_old, df_new])

# Removes header entries
df = df[df['nScenarios'] != 'nScenarios']

# Convert into numbers
df['nScenarios'] = df['nScenarios'].astype(int)
df['PV(kW)'] = df['PV(kW)'].map(lambda entry: entry[1:-1].split())
df['PV(kW)'] = df['PV(kW)'].map(lambda entry: list(np.float_(entry)))
df['Load(kW)'] = df['Load(kW)'].map(lambda entry: entry[1:-1].split())
df['Load(kW)'] = df['Load(kW)'].map(lambda entry: list(np.float_(entry)))
df['sol(keuro)'] = df['sol(keuro)'].astype(float)
df['time(sec)'] = df['time(sec)'].astype(float)
df['memAvg(MB)'] = df['memAvg(MB)'].astype(float)

algorithm_fullname = {
    'DecisionTree_Unbounded': 'Decision Tree (Unbounded Depth)',
    'DecisionTree_MaxDepth10': 'Decision Tree (Maximum Depth: 10)',
    'DecisionTree_MaxDepth15': 'Decision Tree (Maximum Depth: 15)',
    'DecisionTree_MaxDepth20': 'Decision Tree (Maximum Depth: 20)',
	'Ridge': 'Ridge Regression',
	'SVR': 'Support Vector Regression',
    'GradientBoosting': 'Gradient Boosting',
	'RandomForest': 'Random Forest',
    'NN': 'Neural Network'
    }

param_grid = {
          'optimizer' : ['adam'],
          'init' : ['normal'],
          'batch_size' : [200],
          'activation' : ['relu'],
          'hidden_layer_size' : [(100, 100, 100, 100)],
          'epochs' : [100]              
      }

performance_results_data = {
    'target': [],
    'featuresType': [],
    'algorithm': [],
    'features': [],
	'normalization': [],
	'hyperparameters': [],
    'model_params': [],
	'r_2': [],
    'mse': [],
    'mae': [],
    'ev': [],
    'ape': [],
    'trainingTime': []
    }

# #### Util functions

def avg_prediction_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    error = np.sum(np.divide(np.subtract(y_true, y_pred), y_true, out=np.zeros_like(np.subtract(y_true, y_pred)), where=y_true!=0)) / y_true.shape[0]
    return error

def train_and_test(features, target, df=df, max_depth=None, plot=False, normalization='', error_plot=False, CV=False, save_path=None, algorithm="DecisionTree"):
    print(f"#### Features: {features} \n#### Target {target} \n")
    
    if normalization == 'minmax':
        if len(features) > 1:
            df = df[['nScenarios', 'memAvg(MB)', 'time(sec)', 'sol(keuro)', 'PV_mean', 'Load_mean', 'PV_std', 'Load_std']]
        else:
            df = df[['nScenarios', 'memAvg(MB)', 'time(sec)', 'sol(keuro)']]
        df = (df - df.min()) / (df.max() - df.min())
    
    # data prep
    y = df[target]
    X = df[features[0]].apply(lambda x: [x])

    for feat in range(1, len(features)):
        X = X + df[features[feat]].apply(lambda x: [x])
    X = X.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    elapsed_time = -1
    # train
    print("---- Training...")
    if algorithm == "DecisionTree":
        if not CV:
            model = DecisionTreeRegressor(random_state=1, max_depth=max_depth)
            elapsed_time = time.time()
            print(f"{model.fit(X_train, y_train)}\n")
            elapsed_time = time.time() - elapsed_time
            depth = model.get_depth()
            print(f"depth: {depth}\n")
        else:
            model = DecisionTreeRegressor(random_state=1, max_depth=max_depth)
            elapsed_time = time.time() - elapsed_time
            print(f"{model.fit(X_train, y_train)}\n")
            elapsed_time = time.time()
            depth = model.get_depth()
            print(f"Depth with {max_depth} max depth: {depth}")
          
            parameters = {'max_depth':range(6, depth)}
            clf = GridSearchCV(DecisionTreeRegressor(), parameters, n_jobs=4)
            clf.fit(X=X_train, y=y_train)
            model = clf.best_estimator_
            print(f"CV best depth {clf.best_params_}, score {clf.best_score_}\n") 
    elif algorithm == "RandomForest":
        model = RandomForestRegressor(random_state=1, max_depth=max_depth)
        elapsed_time = time.time()
        print(f"{model.fit(X_train, y_train)}\n")
        elapsed_time = time.time() - elapsed_time
    elif algorithm == "Ridge":
        model = Ridge(random_state=1)
        elapsed_time = time.time()
        print(f"{model.fit(X_train, y_train)}\n")
        elapsed_time = time.time() - elapsed_time
    elif algorithm == "GradientBoosting":
        model = GradientBoostingRegressor(random_state=1)
        elapsed_time = time.time()
        print(f"{model.fit(X_train, y_train)}\n")
        elapsed_time = time.time() - elapsed_time
    elif algorithm == "SVR":
        model = SVR()
        elapsed_time = time.time()
        print(f"{model.fit(X_train, y_train)}\n")
        elapsed_time = time.time() - elapsed_time
    elif algorithm == "NN":
        import tensorflow as tf
        import random
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
        from sklearn.model_selection import GridSearchCV

        seed = 1
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        
        def coeff_determination(y_true, y_pred):
            from tensorflow.keras import backend as K
            SS_res =  K.sum(K.square( y_true-y_pred ))
            SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
            return ( 1 - SS_res/(SS_tot + K.epsilon()) )

        def create_model(optimizer='adam', init='normal', hidden_layer_size=(1,), activation='relu'):
            # create model
            model = Sequential()
            for layer_size in hidden_layer_size:
              model.add(Dense(layer_size, kernel_initializer=init, activation=activation))
            model.add(Dense(1, kernel_initializer=init))
            # Compile model
            model.compile(loss='mse', optimizer=optimizer, metrics=[coeff_determination])
            return model
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        model = KerasRegressor(build_fn=create_model, verbose=0)
        # grid search epochs, batch size and optimizer

        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='r2', n_jobs=4)
        elapsed_time = time.time()
        _ = grid.fit(X_train, y_train)
        elapsed_time = time.time() - elapsed_time
        model = grid.best_estimator_
        print(f"CV best params {grid.best_params_}, score {grid.best_score_}\n")
    else:
        return
    
    algorithm_name = algorithm
    if algorithm == "DecisionTree":
        if max_depth != None:
            algorithm_name = algorithm + "_MaxDepth{}".format(max_depth)
        else:
            algorithm_name = algorithm + "_Unbounded"
        
    #test
    print("---- Testing...")
    y_test_pred = model.predict(X_test)
    prediction_results = pd.DataFrame(data=[X_test, y_test, y_test_pred]).transpose(copy=True)
    prediction_results.columns = ['X_test', 'y_test', 'y_test_pred']
    feature_filename = "tracesOnly"
    if len(features) > 1:
       feature_filename = "tracesAndInstance" 
    
    prediction_results_filename = "test_results_{}_{}_{}".format(algorithm_name, feature_filename, target)
    if normalization == 'minmax':
        prediction_results_filename = prediction_results_filename + "_normalized"
    prediction_results_filename = prediction_results_filename + ".csv"
    
    prediction_results.to_csv(prediction_results_filename, index=False)
    
    # Save graph
    fig = plt.figure()
    plt.style.use('seaborn-white')
    max_value = max(max(y_test), max(y_test_pred))
    plt.scatter(y_test, y_test_pred, c='r', s=12, edgecolors='none', zorder=10)
    plt.plot([0, max_value + max_value*0.1], [0, max_value + max_value*0.1], c='b', ls="--", zorder=1, linewidth=1.5)
    plt.xlabel('Actual {} value'.format(target))
    plt.ylabel('Predicted {} value'.format(target))
    plt.xlim(0, max_value + max_value*0.1)
    plt.ylim(0, max_value + max_value*0.1)
    plt.gca().set_title('{}'.format(algorithm_fullname[algorithm_name]))
    
    figure_name = "./graphs/{}/prediction_error_{}_{}_{}".format(target, algorithm_name, feature_filename, target)
    if normalization == "minmax":
        figure_name = figure_name + "_normalized"
    fig.savefig(figure_name, dpi=200, bbox_inches='tight')
    
    print(f"Model: {model.get_params()}")
    if algorithm == "NN":
        r_2 = sklearn.metrics.r2_score(y_test, y_test_pred)
    else:
        r_2 = model.score(X_test, y_test)
    print(f"R^2: {r_2}")
    mse = sklearn.metrics.mean_squared_error(y_test, y_test_pred)
    print(f"MSE: {mse}")
    mae = sklearn.metrics.mean_absolute_error(y_test, y_test_pred)
    print(f"MAE: {mae}")
    ev = sklearn.metrics.explained_variance_score(y_test, y_test_pred)
    print(f"Explained variance: {ev}")
    ape = avg_prediction_error(y_test, y_test_pred)
    print(f"Average prediction error: {ape}")
    
    #save results in a csv
    # features, target, normalization, model, max depth, scores
    if algorithm == "DecisionTree":
        hyperparameters = f'Depth = {depth} (max = {max_depth})'
        if max_depth == None:
            hyperparameters = f'Unbounded depth = {depth}'
        if CV:
            hyperparameters = f"{hyperparameters}. CV best depth {clf.best_params_}, score {clf.best_score_}"
    elif algorithm == "RandomForest":
        hyperparameters = 'None'
    elif algorithm == "Ridge":
        hyperparameters = 'None'
    elif algorithm == "GradientBoosting":
        hyperparameters = 'None'
    elif algorithm == "SVR":
        hyperparameters = 'None'
    elif algorithm == "NN":
        hyperparameters = 'None'
    else:
        return
    
    model_params = model.get_params()
    
    performance_results_data['features'].append(features)
    performance_results_data['featuresType'].append(feature_filename)
    performance_results_data['normalization'].append(normalization)
    performance_results_data['target'].append(target)
    performance_results_data['algorithm'].append(algorithm)
    performance_results_data['hyperparameters'].append(hyperparameters)
    performance_results_data['model_params'].append(model_params)
    performance_results_data['r_2'].append(r_2)
    performance_results_data['mse'].append(mse)
    performance_results_data['mae'].append(mae)
    performance_results_data['ev'].append(ev)
    performance_results_data['ape'].append(ape)
    performance_results_data['trainingTime'].append(elapsed_time)
    
    #save model
    if save_path:
        save_path = "{}_{}".format(save_path, algorithm_name)
        if normalization == "minmax":
            save_path = save_path + "_normalized"
        if algorithm == "NN":
            save_path = save_path + ".h5"
            model.model.save(save_path)   
        else:
            with open(save_path, "wb") as output_file:
                pickle.dump(model, output_file)    
    
    print(model.get_params())
    
# #### Prepare dictionaries

features_target_memAvg = { 'features' : ['nScenarios'],
                        'target' : 'memAvg(MB)'}
features_target_time = { 'features' : ['nScenarios'],
                        'target' : 'time(sec)'}
features_target_cost = { 'features' : ['nScenarios'],
                        'target' : 'sol(keuro)'}

features_target_memAvg_inputs = { 'features' : ['nScenarios', 'PV_mean', 'PV_std', 'Load_mean', 'Load_std'], 
                        'target' : 'memAvg(MB)'}
features_target_time_inputs = { 'features' : ['nScenarios', 'PV_mean', 'PV_std', 'Load_mean', 'Load_std'], 
                        'target' : 'time(sec)'}
features_target_cost_inputs = { 'features' : ['nScenarios', 'PV_mean', 'PV_std', 'Load_mean', 'Load_std'], 
                        'target' : 'sol(keuro)'}


df_inputs = df
df_inputs['PV_mean'] = df['PV(kW)'].map(lambda entry: np.array(entry).mean())
df_inputs['PV_std'] = df['PV(kW)'].map(lambda entry: np.array(entry).std())
df_inputs['Load_mean'] = df['Load(kW)'].map(lambda entry: np.array(entry).mean())
df_inputs['Load_std'] = df['Load(kW)'].map(lambda entry: np.array(entry).std())
df_inputs.describe()


normalization = 'no'

performance_filename = 'performance_metrics_results.csv'
redux_filename = 'performance_metrics_results_redux.csv'


# The following lines of code generate the models ultimately used for the experimentations
algorithm = "DecisionTree"

train_and_test(features_target_cost_inputs['features'], features_target_cost_inputs['target'], max_depth=10, df=df_inputs, normalization=normalization, save_path='ANTICIPATE_input-cost'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_cost_inputs['features'], features_target_cost_inputs['target'], max_depth=15, df=df_inputs, normalization=normalization, save_path='ANTICIPATE_input-cost'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_cost_inputs['features'], features_target_cost_inputs['target'], max_depth=20, df=df_inputs, normalization=normalization, save_path='ANTICIPATE_input-cost'.format(algorithm), algorithm="DecisionTree")

# The following commented lines generate every other model explored

'''
# ###### Decision Tree (Unbounded) Training

# memAvg
train_and_test(features_target_memAvg['features'], features_target_memAvg['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-memory'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_memAvg_inputs['features'], features_target_memAvg_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-memory'.format(algorithm), algorithm="DecisionTree")
# time
train_and_test(features_target_time['features'], features_target_time['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-time'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_time_inputs['features'], features_target_time_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-time'.format(algorithm), algorithm="DecisionTree")
# cost
train_and_test(features_target_cost['features'], features_target_cost['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-cost'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_cost_inputs['features'], features_target_cost_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-cost'.format(algorithm), algorithm="DecisionTree")

# ###### Decision Tree Training (Bounded 5,10,15)

# In[24]:

# 5
    
# memAvg
train_and_test(features_target_memAvg['features'], features_target_memAvg['target'], max_depth=5, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-memory'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_memAvg_inputs['features'], features_target_memAvg_inputs['target'], max_depth=5, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-memory'.format(algorithm), algorithm="DecisionTree")
# time
train_and_test(features_target_time['features'], features_target_time['target'], max_depth=5, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-time'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_time_inputs['features'], features_target_time_inputs['target'], max_depth=5, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-time'.format(algorithm), algorithm="DecisionTree")
# cost
train_and_test(features_target_cost['features'], features_target_cost['target'], max_depth=5, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-cost'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_cost_inputs['features'], features_target_cost_inputs['target'], max_depth=5, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-cost'.format(algorithm), algorithm="DecisionTree")

# 10
    
# memAvg
train_and_test(features_target_memAvg['features'], features_target_memAvg['target'], max_depth=10, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-memory'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_memAvg_inputs['features'], features_target_memAvg_inputs['target'], max_depth=10, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-memory'.format(algorithm), algorithm="DecisionTree")
# time
train_and_test(features_target_time['features'], features_target_time['target'], max_depth=10, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-time'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_time_inputs['features'], features_target_time_inputs['target'], max_depth=10, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-time'.format(algorithm), algorithm="DecisionTree")
# cost
train_and_test(features_target_cost['features'], features_target_cost['target'], max_depth=10, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-cost'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_cost_inputs['features'], features_target_cost_inputs['target'], max_depth=10, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-cost'.format(algorithm), algorithm="DecisionTree")

# 15
    
# memAvg
train_and_test(features_target_memAvg['features'], features_target_memAvg['target'], max_depth=15, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-memory'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_memAvg_inputs['features'], features_target_memAvg_inputs['target'], max_depth=15, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-memory'.format(algorithm), algorithm="DecisionTree")
# time
train_and_test(features_target_time['features'], features_target_time['target'], max_depth=15, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-time'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_time_inputs['features'], features_target_time_inputs['target'], max_depth=15, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-time'.format(algorithm), algorithm="DecisionTree")
# cost
train_and_test(features_target_cost['features'], features_target_cost['target'], max_depth=15, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-cost'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_cost_inputs['features'], features_target_cost_inputs['target'], max_depth=15, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-cost'.format(algorithm), algorithm="DecisionTree")

# ###### Random Forest Training

# In[24]:
    
algorithm = "RandomForest"

# memAvg
train_and_test(features_target_memAvg['features'], features_target_memAvg['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-memory'.format(algorithm), algorithm="RandomForest")
train_and_test(features_target_memAvg_inputs['features'], features_target_memAvg_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-memory'.format(algorithm), algorithm="RandomForest")
# time
train_and_test(features_target_time['features'], features_target_time['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-time'.format(algorithm), algorithm="RandomForest")
train_and_test(features_target_time_inputs['features'], features_target_time_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-time'.format(algorithm), algorithm="RandomForest")
# cost
train_and_test(features_target_cost['features'], features_target_cost['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-cost'.format(algorithm), algorithm="RandomForest")

train_and_test(features_target_cost_inputs['features'], features_target_cost_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/ANTICIPATE_input-cost_new'.format(algorithm), algorithm="RandomForest")

# ###### Ridge Regression Training

# In[24]:

algorithm = "Ridge"

# memAvg
train_and_test(features_target_memAvg['features'], features_target_memAvg['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-memory'.format(algorithm), algorithm="Ridge")
train_and_test(features_target_memAvg_inputs['features'], features_target_memAvg_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-memory'.format(algorithm), algorithm="Ridge")
# time
train_and_test(features_target_time['features'], features_target_time['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-time'.format(algorithm), algorithm="Ridge")
train_and_test(features_target_time_inputs['features'], features_target_time_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-time'.format(algorithm), algorithm="Ridge")
# cost
train_and_test(features_target_cost['features'], features_target_cost['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-cost'.format(algorithm), algorithm="Ridge")
train_and_test(features_target_cost_inputs['features'], features_target_cost_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-cost'.format(algorithm), algorithm="Ridge")

# ###### Gradient Boosting Training

# In[24]:

algorithm = "GradientBoosting"    

# memAvg
train_and_test(features_target_memAvg['features'], features_target_memAvg['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-memory'.format(algorithm), algorithm="GradientBoosting")
train_and_test(features_target_memAvg_inputs['features'], features_target_memAvg_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-memory'.format(algorithm), algorithm="GradientBoosting")
# time
train_and_test(features_target_time['features'], features_target_time['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-time'.format(algorithm), algorithm="GradientBoosting")
train_and_test(features_target_time_inputs['features'], features_target_time_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-time'.format(algorithm), algorithm="GradientBoosting")
# cost
train_and_test(features_target_cost['features'], features_target_cost['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-cost'.format(algorithm), algorithm="GradientBoosting")
train_and_test(features_target_cost_inputs['features'], features_target_cost_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-cost'.format(algorithm), algorithm="GradientBoosting")

# ###### SVR Training

# In[24]:

algorithm = "SVR"      

# memAvg
train_and_test(features_target_memAvg['features'], features_target_memAvg['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-memory'.format(algorithm), algorithm="SVR")
train_and_test(features_target_memAvg_inputs['features'], features_target_memAvg_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-memory'.format(algorithm), algorithm="SVR")
# time
train_and_test(features_target_time['features'], features_target_time['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-time'.format(algorithm), algorithm="SVR")
train_and_test(features_target_time_inputs['features'], features_target_time_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-time'.format(algorithm), algorithm="SVR")
# cost
train_and_test(features_target_cost['features'], features_target_cost['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-cost'.format(algorithm), algorithm="SVR")
train_and_test(features_target_cost_inputs['features'], features_target_cost_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-cost'.format(algorithm), algorithm="SVR")

# In[18]:

algorithm = "NN"      

# memAvg
train_and_test(features_target_memAvg['features'], features_target_memAvg['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-memory'.format(algorithm), algorithm="NN")
train_and_test(features_target_memAvg_inputs['features'], features_target_memAvg_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-memory'.format(algorithm), algorithm="NN")
# time
train_and_test(features_target_time['features'], features_target_time['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-time'.format(algorithm), algorithm="NN")
train_and_test(features_target_time_inputs['features'], features_target_time_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-time'.format(algorithm), algorithm="NN")
# cost
train_and_test(features_target_cost['features'], features_target_cost['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-cost'.format(algorithm), algorithm="NN")
train_and_test(features_target_cost_inputs['features'], features_target_cost_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-cost'.format(algorithm), algorithm="NN")

# In[18]:
'''
performance_results_df = pd.DataFrame(data=performance_results_data)
performance_results_df.to_csv(performance_filename, index=False)

results_redux_df = performance_results_df.round(3).drop(columns=['features', 'normalization', 'model_params']).sort_values(['target', 'featuresType'])
results_redux_df.to_csv(redux_filename, index=False)
'''
# In[18]:
    
performance_results_data = {
    'target': [],
    'featuresType': [],
    'algorithm': [],
    'features': [],
	'normalization': [],
	'hyperparameters': [],
    'model_params': [],
	'r_2': [],
    'mse': [],
    'ev': [],
    'ape': [],
    'trainingTime': []
    }
    
normalization = 'minmax'

performance_filename = 'performance_metrics_results_normalized.csv'
redux_filename = 'performance_metrics_results_redux_normalized.csv'

# ###### Decision Tree (Unbounded) Training

# In[24]:
    
algorithm = "DecisionTree"

# memAvg
#train_and_test(features_target_memAvg['features'], features_target_memAvg['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-memory'.format(algorithm), algorithm="DecisionTree")
#train_and_test(features_target_memAvg_inputs['features'], features_target_memAvg_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-memory'.format(algorithm), algorithm="DecisionTree")
# time
#train_and_test(features_target_time['features'], features_target_time['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-time'.format(algorithm), algorithm="DecisionTree")
#train_and_test(features_target_time_inputs['features'], features_target_time_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-time'.format(algorithm), algorithm="DecisionTree")
# cost
#train_and_test(features_target_cost['features'], features_target_cost['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-cost'.format(algorithm), algorithm="DecisionTree")
#train_and_test(features_target_cost_inputs['features'], features_target_cost_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-cost'.format(algorithm), algorithm="DecisionTree")
# ###### Decision Tree Training (Bounded 5,10,15)

# In[24]:

# 5
    
# memAvg
train_and_test(features_target_memAvg['features'], features_target_memAvg['target'], max_depth=5, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-memory'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_memAvg_inputs['features'], features_target_memAvg_inputs['target'], max_depth=5, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-memory'.format(algorithm), algorithm="DecisionTree")
# time
train_and_test(features_target_time['features'], features_target_time['target'], max_depth=5, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-time'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_time_inputs['features'], features_target_time_inputs['target'], max_depth=5, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-time'.format(algorithm), algorithm="DecisionTree")
# cost
train_and_test(features_target_cost['features'], features_target_cost['target'], max_depth=5, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-cost'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_cost_inputs['features'], features_target_cost_inputs['target'], max_depth=5, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-cost'.format(algorithm), algorithm="DecisionTree")

# 10
    
# memAvg
train_and_test(features_target_memAvg['features'], features_target_memAvg['target'], max_depth=10, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-memory'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_memAvg_inputs['features'], features_target_memAvg_inputs['target'], max_depth=10, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-memory'.format(algorithm), algorithm="DecisionTree")
# time
train_and_test(features_target_time['features'], features_target_time['target'], max_depth=10, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-time'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_time_inputs['features'], features_target_time_inputs['target'], max_depth=10, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-time'.format(algorithm), algorithm="DecisionTree")
# cost
train_and_test(features_target_cost['features'], features_target_cost['target'], max_depth=10, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-cost'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_cost_inputs['features'], features_target_cost_inputs['target'], max_depth=10, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-cost'.format(algorithm), algorithm="DecisionTree")

# 15
    
# memAvg
train_and_test(features_target_memAvg['features'], features_target_memAvg['target'], max_depth=15, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-memory'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_memAvg_inputs['features'], features_target_memAvg_inputs['target'], max_depth=15, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-memory'.format(algorithm), algorithm="DecisionTree")
# time
train_and_test(features_target_time['features'], features_target_time['target'], max_depth=15, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-time'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_time_inputs['features'], features_target_time_inputs['target'], max_depth=15, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-time'.format(algorithm), algorithm="DecisionTree")
# cost
train_and_test(features_target_cost['features'], features_target_cost['target'], max_depth=15, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-cost'.format(algorithm), algorithm="DecisionTree")
train_and_test(features_target_cost_inputs['features'], features_target_cost_inputs['target'], max_depth=15, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-cost'.format(algorithm), algorithm="DecisionTree")

# ###### Random Forest Training

# In[24]:
    
algorithm = "RandomForest"

# memAvg
train_and_test(features_target_memAvg['features'], features_target_memAvg['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-memory'.format(algorithm), algorithm="RandomForest")
train_and_test(features_target_memAvg_inputs['features'], features_target_memAvg_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-memory'.format(algorithm), algorithm="RandomForest")
# time
train_and_test(features_target_time['features'], features_target_time['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-time'.format(algorithm), algorithm="RandomForest")
train_and_test(features_target_time_inputs['features'], features_target_time_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-time'.format(algorithm), algorithm="RandomForest")
# cost
train_and_test(features_target_cost['features'], features_target_cost['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-cost'.format(algorithm), algorithm="RandomForest")
train_and_test(features_target_cost_inputs['features'], features_target_cost_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-cost'.format(algorithm), algorithm="RandomForest")

# ###### Ridge Regression Training

# In[24]:

algorithm = "Ridge"

# memAvg
train_and_test(features_target_memAvg['features'], features_target_memAvg['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-memory'.format(algorithm), algorithm="Ridge")
train_and_test(features_target_memAvg_inputs['features'], features_target_memAvg_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-memory'.format(algorithm), algorithm="Ridge")
# time
train_and_test(features_target_time['features'], features_target_time['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-time'.format(algorithm), algorithm="Ridge")
train_and_test(features_target_time_inputs['features'], features_target_time_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-time'.format(algorithm), algorithm="Ridge")
# cost
train_and_test(features_target_cost['features'], features_target_cost['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-cost'.format(algorithm), algorithm="Ridge")
train_and_test(features_target_cost_inputs['features'], features_target_cost_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-cost'.format(algorithm), algorithm="Ridge")

# ###### Gradient Boosting Training

# In[24]:

algorithm = "GradientBoosting"    

# memAvg
train_and_test(features_target_memAvg['features'], features_target_memAvg['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-memory'.format(algorithm), algorithm="GradientBoosting")
train_and_test(features_target_memAvg_inputs['features'], features_target_memAvg_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-memory'.format(algorithm), algorithm="GradientBoosting")
# time
train_and_test(features_target_time['features'], features_target_time['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-time'.format(algorithm), algorithm="GradientBoosting")
train_and_test(features_target_time_inputs['features'], features_target_time_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-time'.format(algorithm), algorithm="GradientBoosting")
# cost
train_and_test(features_target_cost['features'], features_target_cost['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-cost'.format(algorithm), algorithm="GradientBoosting")
train_and_test(features_target_cost_inputs['features'], features_target_cost_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-cost'.format(algorithm), algorithm="GradientBoosting")

# ###### SVR Training

# In[24]:

algorithm = "SVR"      

# memAvg
train_and_test(features_target_memAvg['features'], features_target_memAvg['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-memory'.format(algorithm), algorithm="SVR")
train_and_test(features_target_memAvg_inputs['features'], features_target_memAvg_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-memory'.format(algorithm), algorithm="SVR")
# time
train_and_test(features_target_time['features'], features_target_time['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-time'.format(algorithm), algorithm="SVR")
train_and_test(features_target_time_inputs['features'], features_target_time_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-time'.format(algorithm), algorithm="SVR")
# cost
train_and_test(features_target_cost['features'], features_target_cost['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-cost'.format(algorithm), algorithm="SVR")
train_and_test(features_target_cost_inputs['features'], features_target_cost_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-cost'.format(algorithm), algorithm="SVR")
# In[18]:

algorithm = "NN"      

# memAvg
train_and_test(features_target_memAvg['features'], features_target_memAvg['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-memory'.format(algorithm), algorithm="NN")
train_and_test(features_target_memAvg_inputs['features'], features_target_memAvg_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-memory'.format(algorithm), algorithm="NN")
# time
train_and_test(features_target_time['features'], features_target_time['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-time'.format(algorithm), algorithm="NN")
train_and_test(features_target_time_inputs['features'], features_target_time_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-time'.format(algorithm), algorithm="NN")
# cost
train_and_test(features_target_cost['features'], features_target_cost['target'], max_depth=None, df=df, normalization=normalization, save_path='./saved_models/{}/no_input-cost'.format(algorithm), algorithm="NN")
train_and_test(features_target_cost_inputs['features'], features_target_cost_inputs['target'], max_depth=None, df=df_inputs, normalization=normalization, save_path='./saved_models/{}/input-cost'.format(algorithm), algorithm="NN")

# In[18]:

    
performance_results_df = pd.DataFrame(data=performance_results_data)
performance_results_df.to_csv(performance_filename, index=False)

results_redux_df = performance_results_df.round(3).drop(columns=['features', 'normalization', 'model_params']).sort_values(['target', 'featuresType'])
results_redux_df.to_csv(redux_filename, index=False)
'''