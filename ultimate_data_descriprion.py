
from data_processing_multi import data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

data.index = data['Disease']
#data = data[data['Sex'] == 2]

data.drop('Sex', axis=1, inplace=True)
data.drop('Disease', axis=1, inplace=True)


data = data.astype(float)

normalized_data = (data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))

control = normalized_data.loc['Control', :]
btal = normalized_data.loc['BTAL', :]
af = normalized_data.loc['AF', :]
atal = normalized_data.loc['ATAL', :]

control = pd.DataFrame.reset_index(control)
btal = pd.DataFrame.reset_index(btal)
af = pd.DataFrame.reset_index(af)
atal = pd.DataFrame.reset_index(atal)

'''age = pd.concat([af['Age'], btal['Age'], control['Age'], atal['Age']], axis=1)
age = age[:].values
age = pd.DataFrame(age, columns= ['IDA','β-thalassemia','Control', 'α-thalassemia']).assign(Feature='Age')'''

hb = pd.concat([af['Hb'], btal['Hb'], atal['Hb'], control['Hb']], axis=1)
hb = hb[:].values
hb = pd.DataFrame(hb, columns= ['IDA','β-thalassemia', 'α-thalassemia', 'Control']).assign(Feature='Hb')

mcv = pd.concat([af['MCV'], btal['MCV'], atal['MCV'], control['MCV']], axis=1)
mcv = mcv[:].values
mcv = pd.DataFrame(mcv, columns= ['IDA','β-thalassemia', 'α-thalassemia', 'Control']).assign(Feature='MCV')

mch = pd.concat([af['MCH'], btal['MCH'], atal['MCH'], control['MCH']], axis=1)
mch = mch[:].values
mch = pd.DataFrame(mch, columns= ['IDA','β-thalassemia', 'α-thalassemia', 'Control']).assign(Feature='MCH')

rdw = pd.concat([af['RDW'], btal['RDW'], atal['RDW'], control['RDW']], axis=1)
rdw = rdw[:].values
rdw = pd.DataFrame(rdw, columns= ['IDA','β-thalassemia', 'α-thalassemia', 'Control']).assign(Feature='RDW')

'''fe = pd.concat([af['Fe'], btal['Fe'], control['Fe']], axis=1)
fe = fe[:].values
fe = pd.DataFrame(fe, columns= ['IDA','β-thalassemia','Control']).assign(Feature='Fe')

tf = pd.concat([af['Tf'], btal['Tf'], control['Tf']], axis=1)
tf = tf[:].values
tf = pd.DataFrame(tf, columns= ['IDA','β-thalassemia','Control']).assign(Feature='Tf')

ft = pd.concat([af['Ft'], btal['Ft'], control['Ft']], axis=1)
ft = ft[:].values
ft = pd.DataFrame(ft, columns= ['IDA','β-thalassemia','Control']).assign(Feature='Ft')
'''
cdf = pd.concat([hb, mcv, mch, rdw])
mdf = pd.melt(cdf, id_vars=['Feature'], var_name=['Class'])
print(mdf.head())

#for big
'''sns.boxplot(x="Feature", y="value", hue="Class", data=mdf, palette="Paired", showfliers = True)
plt.title('Normalized Features per Class')
plt.legend(loc= 1)'''

# for small
sns.boxplot(x="Feature", y="value", hue="Class", data=mdf, palette="Paired", showfliers = True)
plt.title('Normalized Features per Class')
plt.legend(loc= 1, bbox_to_anchor=(1.13, 1.05), framealpha=0.95)

plt.show()

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Calibri'

with PdfPages('test.pdf') as pdf:
    pdf.savefig()