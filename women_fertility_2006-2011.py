import pandas as pd
import numpy as np
from sklearn import cross_validation, neighbors

df = pd.read_csv('f:/data/fertility-edu-levelwomen2006-2011.csv')
df.drop(['Year'], 1, inplace=True)
df.drop(['Area'], 1, inplace=True)

X = np.array(df.drop(['Educational Level of the Women'],1))
y = np.array(df['Educational Level of the Women'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size = 0.2)

clf = neighbors.KNeighborsClassifier()
clf = clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)


print('''
    Enter the following details in specific order:-
    1)Total Fertility Rate
    2)Fertility Rate at age group 15-19
    3)Fertility Rate of Age Group 20-24
    4)Fertility Rate of Age Group 25-29
    5)Fertility Rate of Age Group 30-34
    6)Fertility Rate of Age Group 35-39
    7)Fertility Rate of Age Group 40-44
    8)Fertility Rate of Age Group 45-49

''')
user_input = np.array([input().split()],dtype = np.float64)

result = clf.predict(user_input)
print(result)


print("With accuracy",accuracy)
