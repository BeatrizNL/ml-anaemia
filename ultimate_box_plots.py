import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from collections import defaultdict


data = pd.read_excel(r'C:\Users\preci\Desktop\Mestrado\2º ano\Tese\\BD_INSEF_Talassemicos daniela_beatriz 02 03 2021[3738].xlsx', sheet_name='BD_INSEF_COM TUDO - idade regia')
data.columns = data.iloc[0]
data = data[1:]
data.rename(columns={'SEX': 'Sex', 'Hemoglobina': 'Hb', 'CONCLUSÃO': 'Disease'}, inplace=True)
data = data[["gene HBB", "Sex", "RBC", "Hb", "MCV", "MCH", "RDW", "MCHC"]]
data["Sex"] = [1 if i == "Female" else 2 for i in data["Sex"]]
data = data.copy().dropna()

beta_zero = pd.concat([data.loc[data['gene HBB'] == 'Cd39'], data.loc[data['gene HBB'] == 'IVS-I-1'],
                       data.loc[data['gene HBB'] == 'Cd15'], data.loc[data['gene HBB'] == 'Cd6(-A)']])
beta_zero['Group'] = 'beta_zero'
beta_zero = beta_zero.set_index('Group')

beta_plus = pd.concat([data.loc[data['gene HBB'] =='IVS-I-6'], data.loc[data['gene HBB'] =='IVS-I-110']])
beta_plus['Group'] = 'beta_plus'
beta_plus = beta_plus.set_index('Group')

beta = pd.concat([beta_zero, beta_plus])

print('beta_zero: ', beta_zero['gene HBB'].value_counts(), 'beta_plus: ', beta_plus['gene HBB'].value_counts())

beta.drop('gene HBB', axis=1, inplace=True)

beta = beta.astype(float)

def maleandfemale(beta):
    normalized_beta = (beta-beta.min(axis=0))/(beta.max(axis=0)-beta.min(axis=0))

    beta_zero_normalized = normalized_beta.loc['beta_zero', :]
    beta_plus_normalized = normalized_beta.loc['beta_plus', :]

    beta_zero_normalized = pd.DataFrame.reset_index(beta_zero_normalized)
    beta_plus_normalized = pd.DataFrame.reset_index(beta_plus_normalized)

    Hb = pd.concat([beta_zero_normalized['Hb'], beta_plus_normalized['Hb']], axis=1)
    Hb = Hb[:].values
    Hb = pd.DataFrame(Hb, columns=['$\mathregular{β^{0}}$', '$\mathregular{β^{+}}$']).assign(Feature='Hb')

    MCV = pd.concat([beta_zero_normalized['MCV'], beta_plus_normalized['MCV']], axis=1)
    MCV = MCV[:].values
    MCV = pd.DataFrame(MCV, columns= ['$\mathregular{β^{0}}$', '$\mathregular{β^{+}}$']).assign(Feature='MCV')

    RDW = pd.concat([beta_zero_normalized['RDW'], beta_plus_normalized['RDW']], axis=1)
    RDW = RDW[:].values
    RDW = pd.DataFrame(RDW, columns= ['$\mathregular{β^{0}}$', '$\mathregular{β^{+}}$']).assign(Feature='RDW')

    MCH = pd.concat([beta_zero_normalized['MCH'], beta_plus_normalized['MCH']], axis=1)
    MCH = MCH[:].values
    MCH = pd.DataFrame(MCH, columns= ['$\mathregular{β^{0}}$', '$\mathregular{β^{+}}$']).assign(Feature='MCH')

    cdf = pd.concat([Hb, MCV, MCH, RDW])
    mdf = pd.melt(cdf, id_vars=['Feature'], var_name=['Class'])

    plot_fem_male = plt.figure(figsize=(6, 4))
    sns.boxplot(x="Feature", y="value", hue="Class", data=mdf, palette="tab10", showfliers = True)
    plt.title('β-thalassemia normalized features')
    plt.legend(loc=1)

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'Calibri'
    with PdfPages('test.pdf') as pdf:
        pdf.savefig()

    return plot_fem_male

''' Dividindo por sex para o Hb'''


def hb_gender(beta):
    # female:
    beta_female = beta.loc[beta['Sex'] == 1]

    beta_female = beta_female.astype(float)
    normalized_beta_female = (beta_female-beta_female.min(axis=0))/(beta_female.max(axis=0)-beta_female.min(axis=0))

    female_beta_zero_normalized = normalized_beta_female.loc['beta_zero', :]
    female_beta_plus_normalized = normalized_beta_female.loc['beta_plus', :]

    female_beta_zero_normalized = pd.DataFrame.reset_index(female_beta_zero_normalized)
    female_beta_plus_normalized = pd.DataFrame.reset_index(female_beta_plus_normalized)

    Hb_female = pd.concat([female_beta_zero_normalized['Hb'], female_beta_plus_normalized['Hb']], axis=1)
    Hb_female = Hb_female[:].values
    Hb_female = pd.DataFrame(Hb_female, columns= ['$\mathregular{β^{0}}$', '$\mathregular{β^{+}}$']).assign(Feature='Hb')

    cdf_female = pd.concat([Hb_female])
    mdf_female = pd.melt(cdf_female, id_vars=['Feature'], var_name=['Class'])


    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Hemoglobin in β-thalassemia')
    sns.boxplot(x="Feature", y="value", hue="Class", data=mdf_female, palette="tab10", showfliers = True, ax=ax[0]).set(title='Female')
    ax[0].legend(loc='lower right')

    #male:

    beta_male = beta.loc[beta['Sex'] == 2]

    beta_male = beta_male.astype(float)

    normalized_beta_male = (beta_male-beta_male.min(axis=0))/(beta_male.max(axis=0)-beta_male.min(axis=0))

    male_beta_zero_normalized = normalized_beta_male.loc['beta_zero', :]
    male_beta_plus_normalized = normalized_beta_male.loc['beta_plus', :]

    male_beta_zero_normalized = pd.DataFrame.reset_index(male_beta_zero_normalized)
    male_beta_plus_normalized = pd.DataFrame.reset_index(male_beta_plus_normalized)

    Hb_male = pd.concat([male_beta_zero_normalized['Hb'], male_beta_plus_normalized['Hb']], axis=1)
    Hb_male = Hb_male[:].values
    Hb_male = pd.DataFrame(Hb_male, columns= ['$\mathregular{β^{0}}$', '$\mathregular{β^{+}}$']).assign(Feature='Hb')

    cdf_male = pd.concat([Hb_male])
    mdf_male = pd.melt(cdf_male, id_vars=['Feature'], var_name=['Class'])


    sns.boxplot(x="Feature", y="value", hue="Class", data=mdf_male, palette="tab10", showfliers = True, ax=ax[1]).set(title='Male')

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'Calibri'
    with PdfPages('test.pdf') as pdf:
        pdf.savefig()

    return fig

def pvalues(beta):
    beta_female = beta.loc[beta['Sex'] == 1]
    beta_female = beta_female.astype(float)
    beta_zero_female = beta_female.loc[beta_female.index == 'beta_zero']
    beta_plus_female = beta_female.loc[beta_female.index == 'beta_plus']

    beta_male = beta.loc[beta['Sex'] == 2]
    beta_male = beta_male.astype(float)
    beta_zero_male = beta_male.loc[beta_male.index == 'beta_zero']
    beta_plus_male = beta_male.loc[beta_male.index == 'beta_plus']

    p_value = defaultdict(lambda: {})
    p_value_female = defaultdict(lambda: {})
    p_value_male = defaultdict(lambda: {})

    features = ['Hb', 'MCV', 'MCH', 'RDW']

    for feature in features:
        p_value[feature] = round(stats.ttest_ind(beta_zero[feature], beta_plus[feature])[1], 3)
        p_value_female[feature] = round(stats.ttest_ind(beta_zero_female[feature], beta_plus_female[feature])[
            1], 3)
        p_value_male[feature] = round(stats.ttest_ind(beta_zero_male[feature], beta_plus_male[feature])[1], 3)

    return pd.DataFrame.from_dict(p_value, orient='index'), pd.DataFrame.from_dict(p_value_female, orient='index'), \
           pd.DataFrame.from_dict(p_value_male, orient='index')


plot_fem_male = maleandfemale(beta)
plot = hb_gender(beta)
p_value, p_value_female, p_value_male = pvalues(beta)

plot_fem_male.show()
plot.show()
