import imp
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

train = fetch_20newsgroups(subset="train",shuffle=True)
test = fetch_20newsgroups(subset="train",shuffle=True)

from sklearn.feature_extraction.text import CountVectorizer
cnt_vect = CountVectorizer()
X_train_tf = cnt_vect.fit_transform(train.data)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train_tf)

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train_tfidf,train.target)

X_test_tf = cnt_vect.fit_transform(test.data)
X_test_tfidf = tfidf.fit_transform(X_test_tf)

predicted = mnb.predict(X_test_tfidf)
actual = test.target

print("ACCURACY SCORE\n")
print(accuracy_score(actual,predicted))
print("\n")
print("CONFUSION MATRIX\n")
print(confusion_matrix(actual,predicted))
print("\n")
print("CLASSIFICATION REPORT\n")
print(classification_report(actual,predicted))
print("\n")

