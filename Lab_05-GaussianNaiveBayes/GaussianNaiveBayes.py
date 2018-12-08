import imp
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

c1,c2,c3,c4,c5,c6,c7,c8,c9 = np.loadtxt("Dataset5.csv",delimiter=',',unpack=True)

x = np.column_stack((c1,c2,c3,c4,c5,c6,c7,c8))
y = c9

#Gaussian Naive Bayes:
gnb = GaussianNB()
gnb.fit(x,y)

predicted = gnb.predict(x)

print("ACCURACY SCORE")
print(accuracy_score(y,predicted)*100)

