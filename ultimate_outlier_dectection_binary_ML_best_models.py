from runs_30_indexes_test import best_indexes_test
from sklearn.model_selection import train_test_split
from data_processing_binary_for_outliers import x_data, y_data
from data_processing_binary import x_data as x
from data_processing_binary import y_data as y
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns
from yellowbrick.regressor import CooksDistance
from matplotlib.backends.backend_pdf import PdfPages

def get_cooks_outliers(y_data, x_data):
    dict = {"BTAL": 0, "AF": 1}
    y_data = np.array([dict[disease] for disease in y_data])

    # Instantiate and fit the visualizer
    visualizer = CooksDistance(title='Cook\'s Distance outlier detection for the β-thalassemia and IDA data')
    visualizer.fit(x_data.astype(float), y_data.astype(float))
    cooks_distance = visualizer.distance_
    outliers = cooks_distance[cooks_distance > visualizer.influence_threshold_].index # visualizer.influence_threshold_ = 4/len(y_data)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'Calibri'
    with PdfPages('test.pdf') as pdf:
        pdf.savefig()
    return visualizer, outliers

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
    GA['ANN'] = x_big[['RDW', 'Sex/MCH', 'Hb*MCV', 'Hb/RDW', 'MCV*MCV', 'MCH*MCH', 'RDW*RDW']]
    GA['LR'] = x_big[['RDW', 'Hb', 'MCV*MCH', 'MCH*MCH']]
    GA['SVM'] = x_big[['Hb', 'MCV', 'MCV/RDW']]
    GA['DT'] = x_big[['RDW', 'Sex*RDW', 'Hb*MCH', 'Hb/MCH', 'Hb*RDW', 'MCH*RDW']]
    GA['NB'] = x_big[['Hb/MCH', 'MCV*MCV', 'RDW*RDW']]
    return GA

def get_misclassifieds_and_all_data_accuracy(model, models_names_GA, models_names, GA, y_data, x_data):
    accuracy_all_data = defaultdict(lambda: {})
    misclassified = defaultdict(lambda: {})
    for name_GA in models_names_GA:
        model[name_GA].fit(GA[name_GA], y_data)
        accuracy_all_data[name_GA] = accuracy_score(y_data, model[name_GA].predict(GA[name_GA]))*100
        misclassified[name_GA] = np.where(y_data != model[name_GA].predict(GA[name_GA]))[0]
    for name in models_names:
        model[name].fit(x_data, y_data)
        accuracy_all_data[name] = accuracy_score(y_data, model[name].predict(x_data))*100
        misclassified[name] = np.where(y_data != model[name].predict(x_data))[0]
    print(accuracy_all_data)
    return misclassified, accuracy_all_data

def join_all_misclassified(misclassified, models_names, models_names_GA):
    all_misclassified = np.array([])
    for name in models_names+models_names_GA:
        all_misclassified = np.concatenate((all_misclassified, misclassified[name]), axis=None)
    all_misclassified = all_misclassified.tolist()
    misclassified_counts = {i: all_misclassified.count(i) for i in all_misclassified}
    keys = list(misclassified_counts)
    for key in keys:
        if misclassified_counts[key] < 2:
            del misclassified_counts[key]
    return misclassified_counts

def get_bad_data(misclassified_counts):
    bad_data = list(map(int, misclassified_counts.keys()))
    return bad_data

def my_models_prediction(GA, x_data, models_names, models_names_GA, model, bad_data):
    prediction_my_models = defaultdict(lambda: [])
    my_models_percentage = pd.DataFrame(columns=['AF', 'BTAL'], index= bad_data)
    for i in bad_data:
        for name_GA in models_names_GA:
            prediction_my_models[i] += [model[name_GA].predict(GA[name_GA])[i]]
        for name in models_names:
            prediction_my_models[i] += [model[name].predict(x_data)[i]]
        unique, counts = np.unique(prediction_my_models[i], return_counts=True)
        counts[0] = (counts[0] / sum(counts)) * 100
        counts[1] = 100 - counts[0]
        my_models_percentage.loc[i] = dict(zip(unique, counts))
    return my_models_percentage

def best_indexes_prediction(bad_data, x_data):
    prediction_indexes = {}
    best_indexes_percentage = pd.DataFrame(columns=['AF', 'BTAL'], index= bad_data)
    for i in bad_data:
        prediction_indexes[i] = best_indexes_test(x_data['MCV'][i], x_data['Hb'][i], x_data['RDW'][i], x_data['MCH'][i])
        best_indexes_percentage.loc[i]['AF'] = ((sum(value == 'AF' for value in prediction_indexes[i].values())) / \
                                               len(prediction_indexes[i].values())) * 100
        best_indexes_percentage.loc[i]['BTAL'] = 100 -  best_indexes_percentage.loc[i]['AF']

    return best_indexes_percentage

def join_in_data_frame(misclassified_counts, bad_data, my_models_percentage, best_indexes_percentage, y_data):
    resume_table = pd.concat([(pd.DataFrame(misclassified_counts.values(), index= misclassified_counts.keys(), columns= ['Times misclassified'])),
                        my_models_percentage,
                        best_indexes_percentage,
                        y_data[bad_data]], axis=1)
    return resume_table.sort_index()

def make_veen_diagram(outliers, bad_data):
    fig3 = plt.figure(figsize=(6, 3.2))
    venn2([set(outliers), set(bad_data)], set_labels = ('Outliers', 'Machine Learning\nmisclassified instances'), set_colors=('tab:blue', 'tab:orange'))
    plt.title('Venn Diagram of the instances that Machine Learning binary\n models misclassified and the outliers detected')
    with PdfPages('test.pdf') as pdf:
        pdf.savefig()

    return fig3

def make_box_plot(x, y ,models_names_GA, models_names, model): #usa os dados por uma ordem diferente
    GA = create_more_features(x)
    misclassified, accuracy_all_data = get_misclassifieds_and_all_data_accuracy(model, models_names_GA, models_names, GA, y, x)
    accuracy = defaultdict(lambda: [])
    for name in  models_names_GA:
        for partição in range(30):
            GA_train, GA_test, y_train, y_test = train_test_split(GA[name], y, test_size=0.33, random_state=partição)
            model[name].fit(GA_train, y_train)
            accuracy[name] += [accuracy_score(y_test, model[name].predict(GA_test))*100]

    for name in  models_names:
        for partição in range(30):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=partição)
            model[name].fit(x_train, y_train)
            accuracy[name] += [accuracy_score(y_test, model[name].predict(x_test))*100]

    box = {}
    for name in models_names_GA+models_names:
        box[name] = pd.DataFrame(accuracy[name], columns=['Accuracy'])
        box[name]['Model'] = name

    result_box = pd.concat(box)

    new_accuracy_all_data = pd.DataFrame({'Model': models_names_GA+models_names, 'Accuracy': accuracy_all_data.values()})

    box_plot = plt.figure(figsize=(7, 5))
    #box_plot = plt.figure(figsize=(4, 3.7))
    sns.boxplot(x='Model', y='Accuracy', data=result_box, palette="Set2").set_title('Machine Learning binary models \naccuracy in 30 different splits')
    sns.swarmplot(x='Model', y='Accuracy', data=new_accuracy_all_data, color="red", label="Accuracy with \nall data")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc=3, frameon=True)
    plt.grid(True)
    with PdfPages('test.pdf') as pdf:
        pdf.savefig()

    return box_plot

model = defaultdict(lambda: {})
models_names_GA = ['LR', 'RF', 'ANN', 'NB', 'DT']
models_names = ['SVM']

model['LR'] = LogisticRegression(random_state=0) #GA
model['RF'] = RandomForestClassifier(random_state=0) #GA
model['ANN'] = MLPClassifier(activation='tanh', hidden_layer_sizes=(300, 5), max_iter=300, random_state=4) #GA
model['SVM'] = svm.SVC(C=1.3000000000000003, decision_function_shape='ovo', degree=4, kernel='linear', max_iter=300, probability=True, random_state=0, shrinking=False)
model['NB'] = GaussianNB() #GA
model['DT'] = DecisionTreeClassifier(max_features='sqrt', random_state=17) #GA



visualizer, outliers = get_cooks_outliers(y_data, x_data)
GA = create_more_features(x_data)
misclassified, accuracy_all_data = get_misclassifieds_and_all_data_accuracy(model, models_names_GA, models_names, GA, y_data, x_data)
misclassified_counts = join_all_misclassified(misclassified, models_names, models_names_GA)
bad_data = get_bad_data(misclassified_counts)
my_models_percentage = my_models_prediction(GA, x_data, models_names, models_names_GA, model, bad_data)
best_indexes_percentage = best_indexes_prediction(bad_data, x_data)
resume_table = join_in_data_frame(misclassified_counts, bad_data, my_models_percentage, best_indexes_percentage, y_data)
veen_diagram = make_veen_diagram(outliers, bad_data)
box_plot = make_box_plot(x, y ,models_names_GA, models_names, model)


visualizer.show()
veen_diagram.show()
box_plot.show()
print(resume_table)