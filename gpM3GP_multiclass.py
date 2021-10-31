import pandas as pd
from data_processing_multi import x_data, y_data
from m3gp.M3GP import M3GP
import numpy as np
from m3gp.Constants import *
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=0)

# Train a model
m3gp = M3GP()
m3gp.fit(x_train, y_train)

# Predict test results
pred = m3gp.predict(x_test)
pred = np.array(pred)
# Obtain test accuracy
print( accuracy_score(pred, y_test) )
print(pd.crosstab(y_test, pred, rownames=['Actual Disease'], colnames=['Predicted Disease']))

print(m3gp)
#print('2 D')




