from sklearn.model_selection import train_test_split
from data_processing_multi import x_data, y_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
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

    GA['RF'] = x_big[['Hb', 'Sex/Hb', 'Sex*MCH', 'Hb*Hb', 'Hb/MCH', 'MCV*RDW']]
    GA['NN'] = x_big[['Hb*Hb', 'MCH*MCH', 'RDW*RDW']]
    GA['KNN'] = x_big[['Hb*MCH', 'Hb*RDW', 'MCH*RDW']]
    GA['DT'] = x_big[['Sex*MCH', 'Sex/MCH', 'Hb/MCH', 'MCH/RDW']]
    GA['NB'] = x_big[['Sex*Sex', 'Hb/RDW', 'MCV/MCH', 'MCH*MCH']]


    return GA

def create_models(nstates, models_names, types_with_GA, types_without_GA):
    model = defaultdict(lambda: {})
    for name in models_names:
        for type in types_with_GA+types_without_GA:
            model[name][type] = {state: [] for state in range(nstates)}
    for state in range(nstates):

        model['RF']['basic'][state] = RandomForestClassifier(random_state=state)  # 0
        model['RF']['GA_basic'][state] = RandomForestClassifier(random_state=state)  # 11
        model['RF']['GA_op'][state] = RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=77, n_estimators=200, random_state=state)
        model['RF']['op'][state] = RandomForestClassifier(criterion='entropy', max_depth=89, max_features='log2', min_samples_split=9, n_estimators=250, random_state=state)

        model['NN']['basic'][state] = MLPClassifier(random_state=state)
        model['NN']['GA_basic'][state] = MLPClassifier(random_state=state)
        model['NN']['GA_op'][state] = MLPClassifier(activation='tanh', hidden_layer_sizes=(300, 5), max_iter=300, random_state=state)
        model['NN']['op'][state] = MLPClassifier(activation='identity', max_iter=400, random_state=state, solver='lbfgs')

        model['KNN']['basic'][state] = KNeighborsClassifier()
        model['KNN']['GA_basic'][state] = KNeighborsClassifier()
        model['KNN']['GA_op'][state] = KNeighborsClassifier(algorithm='kd_tree', leaf_size=10, n_neighbors=9, weights='distance')
        model['KNN']['op'][state] = KNeighborsClassifier(algorithm='ball_tree', leaf_size=35, n_neighbors=4, weights='distance')

        model['DT']['basic'][state] = DecisionTreeClassifier(random_state=state)
        model['DT']['GA_basic'][state] = DecisionTreeClassifier(random_state=state)
        model['DT']['GA_op'][state] = DecisionTreeClassifier(max_features='log2', random_state=state)
        model['DT']['op'][state] = DecisionTreeClassifier(criterion='entropy', max_features='log2', random_state=state)

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


def accuracy_per_class(model, y_test, x_test):
    ct = pd.crosstab(y_test, model.predict(x_test), rownames=['Actual Disease'], colnames=['Predicted Disease'])
    accuracy_class = {}
    for i in ct.index:
        if i not in ct.columns:
            ct[i] = 0
        ct_sem_class = ct.drop(i, axis=1).drop(i, axis=0)
        accuracy_class[i] = ((ct[i][i] + ct_sem_class.sum().sum()) / ct.sum().sum())*100
    #accuracy_class = pd.DataFrame.from_dict(accuracy_class, orient='index', columns=['Accuracy'])
    return accuracy_class

def accuracy_part(model, x_train, y_train, x_test, y_test, n_partitions, name, i, state):
    accu = []
    accu_class = []
    for part in range(n_partitions):
        x_train_part, y_train_part, x_test_part, y_test_part = x_train[part], y_train[part], x_test[part], y_test[part]
        accu.append(accuracy_score(y_test_part, model[name][i][state].fit(x_train_part, y_train_part).predict(x_test_part))*100)
        accu_class.append(accuracy_per_class(model[name][i][state], y_test_part, x_test_part))
    keys = accu_class[0].keys()
    accu_class_median = {i: round(np.median([x[i] for x in accu_class]), 1) for i in keys}
    return round(np.median(accu), 1), accu_class_median
    # accu_class_median = {i: round(np.mean([x[i] for x in accu_class]), 1) for i in keys}
    # return round(np.mean(accu), 1), accu_class_median

def accuracy_part_GA(model, x_train_GA, y_train, x_test_GA, y_test, n_partitions, name, i, state):
    accu = []
    accu_class = []
    for part in range(n_partitions):
        x_train_part, y_train_part, x_test_part, y_test_part = x_train_GA[name][part], y_train[part], x_test_GA[name][part], y_test[part]
        accu.append(accuracy_score(y_test_part, model[name][i][state].fit(x_train_part, y_train_part).predict(x_test_part))*100)
        accu_class.append(accuracy_per_class(model[name][i][state], y_test_part, x_test_part))
    keys = accu_class[0].keys()
    accu_class_median = {i: round(np.median([x[i] for x in accu_class]), 1) for i in keys}
    return round(np.median(accu), 1), accu_class_median
    # accu_class_median = {i: round(np.mean([x[i] for x in accu_class]), 1) for i in keys}
    # return round(np.mean(accu), 1), accu_class_median

def confusion_part(model, x_train, y_train, x_test, y_test, n_partitions, name, i, state):
    conf = []
    for part in range(n_partitions):
        x_train_part, y_train_part, x_test_part, y_test_part = x_train[part], y_train[part], x_test[part], y_test[part]
        conf.append(confusion_matrix(y_test_part, model[name][i][state].predict(x_test_part)).ravel())
    return np.median(conf, axis=0)

def confusion_part_GA(model, x_train_GA, y_train, x_test_GA, y_test, n_partitions, name, i, state):
    conf = []
    for part in range(n_partitions):
        x_train_part, y_train_part, x_test_part, y_test_part = x_train_GA[name][part], y_train[part], x_test_GA[name][part], y_test[part]
        conf.append(confusion_matrix(y_test_part, model[name][i][state].predict(x_test_part)).ravel())
    return np.median(conf, axis=0)

def calculate_accuracy_and_confusion_matrix(model, models_names, types_without_GA, types_with_GA, GA_train, GA_test, x_train, x_test, y_train, y_test, true_false, nstates, n_partitions):
    accuracy= defaultdict(lambda: defaultdict(lambda: {}))
    accuracy_class = defaultdict(lambda: defaultdict(lambda: {}))
    confusion = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {})))
    for name in models_names:
        for i in types_without_GA:
            for state in range(nstates):
                accuracy[name][i][state], accuracy_class[name][i][state] = accuracy_part(model, x_train, y_train, x_test, y_test, n_partitions, name, i, state)
                conf = confusion_part(model, x_train, y_train, x_test, y_test, n_partitions, name, i, state)
                for index, tf in enumerate(true_false):
                    confusion[name][i][state][tf] = conf[index]
        for i in types_with_GA:
            for state in range(nstates):
                accuracy[name][i][state], accuracy_class[name][i][state] = accuracy_part_GA(model, GA_train, y_train, GA_test, y_test, n_partitions, name, i,state)
                conf = confusion_part_GA(model, GA_train, y_train, GA_test, y_test, n_partitions, name, i, state)
                for index, tf in enumerate(true_false):
                    confusion[name][i][state][tf] = conf[index]
    return accuracy, accuracy_class, confusion

def get_only_best_states(accuracy, accuracy_class, confusion, models_names, types_with_GA, types_without_GA):
    for name in models_names:
        for i in types_without_GA + types_with_GA:
            max_key = max(accuracy[name][i], key=accuracy[name][i].get)
            all_keys = accuracy[name][i].keys()
            for key in list(all_keys):
                if key != max_key:
                    del accuracy[name][i][key], accuracy_class[name][i][key], confusion[name][i][key]
    return accuracy, accuracy_class, confusion

def transform_in_dataframe(accuracy, accuracy_class, confusion):
    for name in models_names:
        for i in types_with_GA + types_without_GA:
            keys = confusion[name][i].keys()
            confusion[name][i] = [confusion[name][i][k] for k in keys]
            keys_2 = accuracy_class[name][i].keys()
            accuracy_class[name][i] = [accuracy_class[name][i][k] for k in keys_2]
    return pd.DataFrame(accuracy), pd.DataFrame(accuracy_class), pd.DataFrame(confusion)

def run_all_models_and_get_accuracy(x_data, models_names, types_without_GA, types_with_GA, nstates, n_partitions, y_data):
    GA = create_more_features(x_data)
    model = create_models(nstates, models_names, types_with_GA, types_without_GA)
    GA_train, GA_test, x_train, x_test, y_train, y_test = create_different_data_particions(n_partitions, x_data, GA, y_data, models_names)
    accuracy, accuracy_class, confusion = calculate_accuracy_and_confusion_matrix(model, models_names, types_without_GA, types_with_GA, GA_train, GA_test,
                                                                  x_train, x_test, y_train, y_test, true_false, nstates, n_partitions)
    accuracy, accuracy_class, confusion = get_only_best_states(accuracy, accuracy_class, confusion, models_names, types_with_GA, types_without_GA)
    accuracy, accuracy_class, confusion = transform_in_dataframe(accuracy, accuracy_class, confusion)
    return accuracy, accuracy_class, confusion



models_names= ['RF', 'NN', 'KNN', 'DT', 'NB']
types_without_GA = ['basic', 'op']
types_with_GA = ['GA_basic', 'GA_op']

true_false = ['TN', 'FP', 'FN', 'TP']
nstates = 50
n_partitions = 30

accuracy, accuracy_class, confusion = run_all_models_and_get_accuracy(x_data, models_names, types_without_GA, types_with_GA, nstates, n_partitions, y_data)

print('com as novas optimizações')