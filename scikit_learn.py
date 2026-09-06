from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd 

df = pd.read_csv('winequality-red.csv', sep=';')

X = df.drop(columns='quality')
y = df['quality']

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = GaussianNB()

model.fit(x_train,y_train)

predictions = model.predict(x_test)

accuracy = accuracy_score(predictions,y_test)

print('scikit Accuracy:', accuracy)
