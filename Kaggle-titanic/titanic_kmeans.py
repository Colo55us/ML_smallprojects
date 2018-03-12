import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans
#print(df.head())
df = pd.read_excel('F:/Data/test_data.xlsx')
df.drop(['Track Name'],1 , inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)

def handle_non_numeric(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            
            df[column] = list(map(convert_to_int,df[column]))

    return df

df = handle_non_numeric(df)


X = np.array(df.drop(['duration_ms','time_signature','Dancebility'],1).astype(float))
y = np.array(df['Manual Mood Classification'])
X = preprocessing.scale(X)
clf = KMeans(n_clusters = 3)
clf.fit(X)

correct_count = 0
for i in range(len(X)):
    predict_data = np.array(X[i].astype(float))
    predict_data = predict_data.reshape(-1, len(predict_data))
    prediction = clf.predict(predict_data)
    if prediction == y[i]:
        correct_count += 1
print(df.head())
print(correct_count/len(X))



