from sklearn.datasets import load_iris
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import classification_report,  accuracy_score

# Load the dataset
iris = load_iris()

# See it as a table
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print(df)
## data split 
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train.shape)
print(X_test.shape)
## scaling data 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
## training the model

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))


