import pandas as pd
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score
from data_processing_binary import x_data, y_data
from collections import defaultdict
from sklearn.model_selection import train_test_split

from sklearn.model_selection import HalvingRandomSearchCV

def create_more_features(x_data):
    features = x_data.columns
    x_big = x_data.copy()
    for index, i in enumerate(features):
        for e in features[index:]:
            x_big[i + "*" + e] = x_big[i] * x_big[e]
            if i != e:
                x_big[i + "/" + e] = x_big[i] / x_big[e]
    return x_big


def train_test(models_names, x_data):
    x_data_novo = create_more_features(x_data)

    GA = defaultdict(lambda: pd.DataFrame())
    GA['RF'] = x_data_novo[['Hb/MCH', 'MCV*RDW']]
    GA['NN'] = x_data_novo[['RDW', 'Sex/MCH', 'Hb*MCV', 'Hb/RDW', 'MCV*MCV', 'MCH*MCH', 'RDW*RDW']]
    GA['LG'] = x_data_novo[['RDW', 'Hb', 'MCV*MCH', 'MCH*MCH']]
    GA['SVM'] = x_data_novo[['Hb', 'MCV', 'MCV/RDW']]
    GA['DT'] = x_data_novo[['RDW', 'Sex*RDW', 'Hb*MCH', 'Hb/MCH', 'Hb*RDW', 'MCH*RDW']]
    GA['NB'] = x_data_novo[['Hb/MCH', 'MCV*MCV', 'RDW*RDW']]

    GA_train = defaultdict(lambda: pd.DataFrame())
    GA_test = defaultdict(lambda: pd.DataFrame())

    for i in models_names:
        GA_train[i], GA_test[i], y_train, y_test = train_test_split(GA[i], y_data, test_size=0.33, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=0)
    return GA_train, GA_test, x_train, x_test, y_train, y_test


#new optimazed models:
def create_new_optmized_models(param_grid, models, models_names, x_data):
    GA_train, GA_test, x_train, x_test, y_train, y_test = train_test(models_names, x_data)
    for name in models_names:
        models['optimized_models_GA'][name] = HalvingRandomSearchCV(estimator=models['estimators'][name],
                                            param_distributions=param_grid[name], random_state=0).fit(GA_train[name], y_train)
        models['optimized_models'][name] = HalvingRandomSearchCV(estimator=models['estimators'][name],
                                            param_distributions=param_grid[name], random_state=0).fit(x_train, y_train)

    return models['optimized_models'],models['optimized_models_GA'], GA_train, GA_test, x_train, x_test, y_test, y_train

# test the models:
def evaluate_all_models(param_grid, models, models_names, x_data):
    accuracy = defaultdict(lambda: {})
    models['optimized_models'], models['optimized_models_GA'], GA_train, GA_test, x_train, x_test, y_test, y_train = create_new_optmized_models(param_grid, models, models_names, x_data)
    for name in models_names:
        accuracy['optimized_models_GA'][name] = accuracy_score(y_test, models['optimized_models_GA'][name].fit(GA_train[name], y_train).predict(GA_test[name]))
        accuracy['optimized_models'][name] = accuracy_score(y_test,models['optimized_models'][name].fit(x_train, y_train).predict(x_test))
    return pd.DataFrame(accuracy), models['optimized_models'], models['optimized_models_GA']


# param grids:
param_grid = {}

param_grid['RF'] = {"n_estimators": np.arange(50,300,50),
              "max_features": ["auto", "sqrt", "log2"],
              "min_samples_split": np.arange(2,10,1),
              "bootstrap": [True, False],
                'max_depth': np.arange(50,100,1),
              "criterion": ["gini", "entropy"]}

param_grid['LG'] = {"C": np.arange(0.1,1.5,0.1),
              "max_iter": np.arange(100, 500, 100),
              'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'fit_intercept':[True,False]}

param_grid['NN'] = {'solver':['lbfgs', 'sgd', 'adam'],
              'activation': ['identity', 'logistic', 'tanh', 'relu'],
             'hidden_layer_sizes': [(100,), (300, 5), (400, 6), (250, 6), (500, 6), (300,2)],
            'max_iter':np.arange(100, 1000, 100)}

param_grid['SVM'] = {'C': np.arange(0.1, 1.5, 0.1),
                     "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                      "degree": np.arange(1, 6, 1),
                     'gamma': ['scale', 'auto'],
                     'shrinking':[True, False],
                     'max_iter': np.append(np.arange(100, 500, 100,), -1),
                     'decision_function_shape': ['ovo', 'ovr']}

param_grid['DT'] = {"criterion":['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    "max_features": ['auto', 'sqrt', 'log2']}

param_grid['NB'] = {"var_smoothing":[1e-11, 1e-10, 1e-9, 1e-8, 1e-7]}

models = defaultdict(lambda: {})
#deflaut models:
models['estimators']['RF'] = RandomForestClassifier(random_state=11)
models['estimators']['LG'] = LogisticRegression(random_state=0)
models['estimators']['NN'] = MLPClassifier(random_state=0)
models['estimators']['SVM'] = svm.SVC(probability=True, random_state=0)
models['estimators']['DT'] = DecisionTreeClassifier(random_state=0)
models['estimators']['NB'] = GaussianNB()

models_names = ['RF', 'NN', 'LG', 'SVM', 'DT', 'NB']

accuracy, new_models, new_models_GA = evaluate_all_models(param_grid, models, models_names, x_data)

print(accuracy)