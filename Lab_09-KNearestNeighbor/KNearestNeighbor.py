import imp
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import neighbors

iris = datasets.load_iris()

X = iris.data
Y = iris.target

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.5,random_state=42)

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)
predicted = knn.predict(X_test)

actual = Y_test

print("ACTUAL CLASS")
print(actual)

print("PREDICTED CLASS")
print(predicted)


print("ACCURACY SCORE")
print(accuracy_score(actual,predicted))

print("CONFUSION MATRIX")
print(confusion_matrix(actual,predicted))

print("CLASSIFICATION REPORT")
print(classification_report(actual,predicted))


