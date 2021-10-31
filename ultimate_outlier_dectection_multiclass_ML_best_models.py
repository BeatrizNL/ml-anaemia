from data_processing_multi_for_outliers import x_data, y_data
from data_processing_multi import x_data as x
from data_processing_multi import y_data as y
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns
from yellowbrick.regressor import CooksDistance
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter

def get_cooks_outliers(y_data, x_data):
    dict = {"BTAL": 0, "α-TAL": 1, "AF": 2, "Control": 3}
    y_data = np.array([dict[disease] for disease in y_data])

    # Instantiate and fit the visualizer
    visualizer = CooksDistance(title='Cook\'s Distance outlier detection for all the data')
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

    GA['RF'] = x_big[['Hb', 'Sex/Hb', 'Sex*MCH', 'Hb*Hb', 'Hb/MCH', 'MCV*RDW']]
    GA['ANN'] = x_big[['Hb*Hb', 'MCH*MCH', 'RDW*RDW']]
    GA['KNN'] = x_big[['Hb*MCH', 'Hb*RDW', 'MCH*RDW']]
    GA['DT'] = x_big[['Sex*MCH', 'Sex/MCH', 'Hb/MCH', 'MCH/RDW']]
    GA['NB'] = x_big[['Sex*Sex', 'Hb/RDW', 'MCV/MCH', 'MCH*MCH']]

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
    return misclassified, accuracy_all_data

def join_all_misclassified(misclassified, models_names, models_names_GA):
    all_misclassified = np.array([])
    for name in models_names+models_names_GA:
        all_misclassified = np.concatenate((all_misclassified, misclassified[name]), axis=None)
    all_misclassified = all_misclassified.tolist()
    misclassified_counts = {i: all_misclassified.count(i) for i in all_misclassified}
    keys = list(misclassified_counts)
    for key in keys:
        if misclassified_counts[key] < 3:
            del misclassified_counts[key]
    return misclassified_counts

def get_bad_data(misclassified_counts):
    bad_data = list(map(int, misclassified_counts.keys()))
    return bad_data

def my_models_prediction(bad_data, model, models_names, GA, x_data):
    prediction_my_models = defaultdict(lambda: [])
    my_models_percentage = pd.DataFrame(columns=['AF', 'BTAL', 'α-TAL', 'Control'], index=bad_data)
    for i in bad_data:
        for name in models_names_GA:
            prediction_my_models[i] += [model[name].predict(GA[name])[i]]
        for name in models_names:
            prediction_my_models[i] += [model[name].predict(x_data)[i]]
        unique, counts = np.unique(prediction_my_models[i], return_counts=True)
        counts_new = np.zeros_like(counts)
        for index in range(len(counts)):
            counts_new[index] = (counts[index] / sum(counts)) * 100
        my_models_percentage.loc[i] = dict(zip(unique, counts_new))

    return my_models_percentage

def join_in_data_frame(my_models_percentage, y_data, bad_data, misclassified_counts):
    resume_table = pd.concat([(pd.DataFrame(misclassified_counts.values(), index=misclassified_counts.keys(),
                                            columns=['Times misclassified'])),
                              my_models_percentage,
                              y_data[bad_data]], axis=1)
    return resume_table.sort_index()

def make_veen_diagram(outliers, bad_data):
    fig3 = plt.figure(figsize=(6, 3.2))
    venn2([set(outliers), set(bad_data)], set_labels=('Outliers', 'Machine Learning\nmisclassified instances'),
          set_colors=('red', 'yellow'))
    plt.title(
        'Venn Diagram of the instances that Machine Learning multi-class\n models misclassified and the outliers detected')
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
    box_plot, ax1 = plt.subplots(figsize=(7, 5))
    #box_plot, ax1 = plt.subplots(figsize=(4, 3.7))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    sns.boxplot(x='Model', y='Accuracy', data=result_box, palette="Set3").set_title('Machine Learning multi-class models \naccuracy in 30 different splits')
    sns.swarmplot(x='Model', y='Accuracy', data=new_accuracy_all_data, color="red", label="Accuracy with \nall data")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc=2, frameon=True)
    plt.grid(True)
    with PdfPages('test.pdf') as pdf:
        pdf.savefig()

    return box_plot

model = defaultdict(lambda: {})
models_names= ['RF', 'KNN', 'ANN', 'DT']
models_names_GA= ['NB']

model['RF'] = RandomForestClassifier(criterion='entropy', max_depth=89, max_features='log2', min_samples_split=9, n_estimators=250, random_state=49)
model['ANN'] = MLPClassifier(activation='identity', max_iter=400, random_state=17, solver='lbfgs')
model['KNN'] = KNeighborsClassifier()
model['DT'] = DecisionTreeClassifier(criterion='entropy', max_features='log2', random_state=7)
model['NB'] = GaussianNB() #GA


visualizer, outliers = get_cooks_outliers(y_data, x_data)
GA = create_more_features(x_data)
misclassified, accuracy_all_data = get_misclassifieds_and_all_data_accuracy(model, models_names_GA, models_names, GA, y_data, x_data)
misclassified_counts = join_all_misclassified(misclassified, models_names, models_names_GA)
bad_data = get_bad_data(misclassified_counts)
my_models_percentage = my_models_prediction(bad_data, model, models_names, GA, x_data)
resume_table = join_in_data_frame(my_models_percentage, y_data, bad_data, misclassified_counts)
veen_diagram = make_veen_diagram(outliers, bad_data)
box_plot = make_box_plot(x, y ,models_names_GA, models_names, model)

visualizer.show()
veen_diagram.show()
box_plot.show()
print(resume_table)