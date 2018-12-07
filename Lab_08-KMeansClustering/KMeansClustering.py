import imp

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
#from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing

iris = datasets.load_iris()

X = pd.DataFrame(iris.data)
X.columns = ["Sepal_Length","Sepal_Width","Petal_Length","Petal_Width"]
X_norm = preprocessing.normalize(X)

Y = pd.DataFrame(iris.target)
Y.columns = ["Targets"]

#KMEANS
model = KMeans(n_clusters = 3)
model.fit(X_norm)

#EM model
gmm = GaussianMixture(n_components=3)
gmm.fit(X_norm)
gmm_y = gmm.predict(X_norm)

plt.figure(figsize=(14,14))
colormap = np.array(["blue","red","yellow"])

#plot Real Cluster
plt.subplot(2,2,1)
plt.scatter(X.Petal_Length,X.Petal_Width,c = colormap[Y.Targets],s=40)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("REAL CLUSTER")

#plot KMeans
plt.subplot(2,2,2)
plt.scatter(X.Petal_Length,X.Petal_Width,c = colormap[model.labels_],s=40)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("K-MEANS")

#plot GMM
plt.subplot(2,2,3)
plt.scatter(X.Petal_Length,X.Petal_Width,c = colormap[gmm_y],s=40)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("GMM")

plt.show()
