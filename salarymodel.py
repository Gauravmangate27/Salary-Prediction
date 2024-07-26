
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
dataset=pd.read_csv('/content/Salary_Data.csv')
dataset.head(5)
dataset.info()
dataset.shape
dataset.describe()

x=dataset['YearsExperience']
y=dataset['Salary']
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.scatter(x,y)
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
model = LinearRegression()
x=dataset[['YearsExperience','Salary']]
y=dataset['Salary']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
from sklearn.metrics import confusion_matrix
matrix=confusion_matrix(y_test,y_pred)
print(matrix)
y_pred=logreg.predict(x_test)
print(x_test)
print(y_pred)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('R-squared:', metrics.r2_score(y_test, y_pred))
