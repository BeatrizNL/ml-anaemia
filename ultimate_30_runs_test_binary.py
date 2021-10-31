from sklearn.model_selection import train_test_split
from data_processing_binary import x_data, y_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from collections import defaultdict
import pandas as pd
import numpy as np


def create_more_features(x_data):
    features = x_data.columns
    x_big = x_data.copy()
    for index, i in enumerate(features):
        for e in features[index:]:
            x_big[i + "*" + e] = x_big[i] * x_big[e]
            if i != e:
                x_big[i + "/" + e] = x_big[i] / x_big[e]
    GA = defaultdict(lambda: pd.DataFrame())

    GA['RF'] = x_big[['Hb/MCH', 'MCV*RDW']]
    GA['NN'] = x_big[['RDW', 'Sex/MCH', 'Hb*MCV', 'Hb/RDW', 'MCV*MCV', 'MCH*MCH', 'RDW*RDW']]
    GA['LG'] = x_big[['RDW', 'Hb', 'MCV*MCH', 'MCH*MCH']]
    GA['SVM'] = x_big[['Hb', 'MCV', 'MCV/RDW']]
    GA['DT'] = x_big[['RDW', 'Sex*RDW', 'Hb*MCH', 'Hb/MCH', 'Hb*RDW', 'MCH*RDW']]
    GA['NB'] = x_big[['Hb/MCH', 'MCV*MCV', 'RDW*RDW']]
    return GA

def create_models(nstates, models_names, types_with_GA, types_without_GA):
    model = defaultdict(lambda: {})
    for name in models_names:
        for type in types_with_GA+types_without_GA:
            model[name][type] = {state: [] for state in range(nstates)}
    for state in range(nstates):
        model['LG']['basic'][state] = LogisticRegression(random_state=state)  # 0
        model['LG']['GA_basic'][state] = LogisticRegression(random_state=state)  # 0
        model['LG']['GA_op'][state] = LogisticRegression(C=1.2000000000000002, random_state=state, solver='newton-cg')
        model['LG']['op'][state] = LogisticRegression(C=0.6, fit_intercept=False, max_iter=300, random_state=state, solver='sag')

        model['RF']['basic'][state] = RandomForestClassifier(random_state=state)  # 0
        model['RF']['GA_basic'][state] = RandomForestClassifier(random_state=state)  # 11
        model['RF']['GA_op'][state] = RandomForestClassifier(bootstrap=False, max_depth=59, max_features='sqrt', n_estimators=200, random_state=state)
        model['RF']['op'][state] = RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=77, n_estimators=200, random_state=state)

        model['NN']['basic'][state] = MLPClassifier(random_state=state)
        model['NN']['GA_basic'][state] = MLPClassifier(random_state=state)
        model['NN']['GA_op'][state] = MLPClassifier(activation='tanh', hidden_layer_sizes=(300, 5), max_iter=300, random_state=state)
        model['NN']['op'][state] = MLPClassifier(activation='tanh', hidden_layer_sizes=(300, 5), max_iter=300, random_state=state)

        model['SVM']['basic'][state] = svm.SVC(probability=True, random_state=state)
        model['SVM']['GA_basic'][state] = svm.SVC(probability=True, random_state=state)
        model['SVM']['GA_op'][state] = svm.SVC(C=1.3000000000000003, decision_function_shape='ovo', degree=4, kernel='linear', max_iter=300, probability=True, random_state=state, shrinking=False)
        model['SVM']['op'][state] = svm.SVC(C=1.3000000000000003, decision_function_shape='ovo', degree=4, kernel='linear', max_iter=300, probability=True, random_state=state, shrinking=False)

        model['DT']['basic'][state] = DecisionTreeClassifier(random_state=state)
        model['DT']['GA_basic'][state] = DecisionTreeClassifier(random_state=state)
        model['DT']['GA_op'][state] = DecisionTreeClassifier(max_features='sqrt', random_state=state)
        model['DT']['op'][state] = DecisionTreeClassifier(criterion='entropy', max_features='sqrt', random_state=state)

        model['NB']['basic'][state] = GaussianNB()
        model['NB']['GA_basic'][state] = GaussianNB()
        model['NB']['GA_op'][state] = GaussianNB(var_smoothing=1e-08)
        model['NB']['op'][state] = GaussianNB(var_smoothing=1e-08)
    return model

def create_different_data_particions(n_partitions, x_data, GA, y_data, models_names):
    GA_train = {}
    GA_test = {}
    x_train = {}
    x_test = {}
    y_train = {}
    y_test = {}
    for name in models_names:
        GA_train[name] = defaultdict(lambda: {})
        GA_test[name] = defaultdict(lambda: {})
        for part in range(n_partitions):
            GA_train[name][part] = pd.DataFrame()
            GA_test[name][part] = pd.DataFrame()
            x_train[part] = defaultdict(lambda: pd.DataFrame())
            x_test[part] = defaultdict(lambda: pd.DataFrame())
            y_train[part] = defaultdict(lambda: pd.DataFrame())
            y_test[part] = defaultdict(lambda: pd.DataFrame())
    for part in range(n_partitions):
        x_train[part], x_test[part], y_train[part], y_test[part] = train_test_split(x_data, y_data, test_size=0.33, random_state=part)
        for name in models_names:
            GA_train[name][part], GA_test[name][part], y_train[part], y_test[part] = train_test_split(GA[name], y_data, test_size=0.33, random_state=part)

    return GA_train, GA_test, x_train, x_test, y_train, y_test


def accuracy_part_GA(model, x_train_GA, y_train, x_test_GA, y_test, n_partitions, name, i, state):
    accu = []
    for part in range(n_partitions):
        x_train_part, y_train_part, x_test_part, y_test_part = x_train_GA[name][part], y_train[part], x_test_GA[name][part], y_test[part]
        accu.append(accuracy_score(y_test_part, model[name][i][state].fit(x_train_part, y_train_part).predict(x_test_part))*100)
    return round(np.median(accu), 1)

def confusion_part_GA(model, x_train_GA, y_train, x_test_GA, y_test, n_partitions, name, i, state):
    conf = []
    for part in range(n_partitions):
        x_train_part, y_train_part, x_test_part, y_test_part = x_train_GA[name][part], y_train[part], x_test_GA[name][part], y_test[part]
        conf.append(confusion_matrix(y_test_part, model[name][i][state].predict(x_test_part)).ravel())
    return np.median(conf, axis=0)

def accuracy_part(model, x_train, y_train, x_test, y_test, n_partitions, name, i, state):
    accu = []
    for part in range(n_partitions):
        x_train_part, y_train_part, x_test_part, y_test_part = x_train[part], y_train[part], x_test[part], y_test[part]
        accu.append(accuracy_score(y_test_part, model[name][i][state].fit(x_train_part, y_train_part).predict(x_test_part))*100)
    return round(np.median(accu), 1)

def confusion_part(model, x_train, y_train, x_test, y_test, n_partitions, name, i, state):
    conf = []
    for part in range(n_partitions):
        x_train_part, y_train_part, x_test_part, y_test_part = x_train[part], y_train[part], x_test[part], y_test[part]
        conf.append(confusion_matrix(y_test_part, model[name][i][state].predict(x_test_part)).ravel())
    return np.median(conf, axis=0)

def calculate_accuracy_and_confusion_matrix(model, models_names, types_without_GA, types_with_GA, GA_train, GA_test, x_train, x_test, y_train, y_test, true_false, nstates, n_partitions):
    accuracy= defaultdict(lambda: defaultdict(lambda: {}))
    confusion = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {})))
    for name in models_names:
        for i in types_without_GA:
            for state in range(nstates):
                accuracy[name][i][state] = accuracy_part(model, x_train, y_train, x_test, y_test, n_partitions, name, i, state)
                conf = confusion_part(model, x_train, y_train, x_test, y_test, n_partitions, name, i, state)
                for index, tf in enumerate(true_false):
                    confusion[name][i][state][tf] = conf[index]
        for i in types_with_GA:
            for state in range(nstates):
                accuracy[name][i][state] = accuracy_part_GA(model, GA_train, y_train, GA_test, y_test, n_partitions, name, i, state)
                conf = confusion_part_GA(model, GA_train, y_train, GA_test, y_test, n_partitions, name, i, state)
                for index, tf in enumerate(true_false):
                    confusion[name][i][state][tf] = conf[index]
    return accuracy, confusion

def get_only_best_states(accuracy, confusion, models_names, types_with_GA, types_without_GA):
    for name in models_names:
        for i in types_with_GA + types_without_GA:
            max_key = max(accuracy[name][i], key=accuracy[name][i].get)
            all_keys = accuracy[name][i].keys()
            for key in list(all_keys):
                if key != max_key:
                    del accuracy[name][i][key], confusion[name][i][key]
    return accuracy, confusion

def transform_in_dataframe(accuracy, confusion):
    for name in models_names:
        for i in types_with_GA + types_without_GA:
            keys = confusion[name][i].keys()
            confusion[name][i] = [confusion[name][i][k] for k in keys]

    return pd.DataFrame(accuracy), pd.DataFrame(confusion)

def run_all_models_and_get_accuracy(x_data, models_names, types_without_GA, types_with_GA, nstates, n_partitions, y_data):
    GA = create_more_features(x_data)
    model = create_models(nstates, models_names, types_with_GA, types_without_GA)
    GA_train, GA_test, x_train, x_test, y_train, y_test = create_different_data_particions(n_partitions, x_data, GA, y_data, models_names)
    accuracy, confusion = calculate_accuracy_and_confusion_matrix(model, models_names, types_without_GA, types_with_GA, GA_train, GA_test,
                                                                  x_train, x_test, y_train, y_test, true_false, nstates, n_partitions)
    accuracy, confusion = get_only_best_states(accuracy, confusion, models_names, types_with_GA, types_without_GA)
    accuracy, confusion = transform_in_dataframe(accuracy, confusion)
    return accuracy, confusion



models_names= ['LG', 'RF', 'NN', 'SVM', 'DT', 'NB']
types_without_GA = ['basic', 'op']
types_with_GA = ['GA_basic', 'GA_op']
true_false = ['TN', 'FP', 'FN', 'TP']
nstates = 50 # 50
n_partitions = 30 # 30

accuracy, confusion = run_all_models_and_get_accuracy(x_data, models_names, types_without_GA, types_with_GA, nstates, n_partitions, y_data)

print('com as novas optimizações')