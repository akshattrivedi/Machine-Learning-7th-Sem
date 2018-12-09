import imp
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

train = fetch_20newsgroups(subset="train",shuffle=True)
test = fetch_20newsgroups(subset="test",shuffle=True)

from sklearn.feature_extraction.text import CountVectorizer
cnt_vect = CountVectorizer()
X_train_tf = cnt_vect.fit_transform(train.data)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train_tf)

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train_tfidf,train.target)

X_test_tf = cnt_vect.transform(test.data)
X_test_tfidf = tfidf.transform(X_test_tf)

predicted = mnb.predict(X_test_tfidf)
actual = test.target

print("ACCURACY SCORE\n")
print(accuracy_score(actual,predicted))
print("\n")
print("CONFUSION MATRIX\n")
print(confusion_matrix(actual,predicted))
print("\n")
print("CLASSIFICATION REPORT\n")
print(classification_report(actual,predicted,target_names=train.target_names))
print("\n")

#EXTRA
from io import StringIO

print("SHAPE OF TRAINING AND TEST DATASETS")
print("Train: ",X_train_tfidf.shape)
print("Test : ",X_test_tfidf.shape)
print()

text = input("Enter the sentence to predict the class: ")
sent = StringIO(text)
sent_tf = cnt_vect.transform(sent)
sent_tfidf = tfidf.transform(sent_tf)
print("Shape(Sentence): ",sent_tfidf.shape)

print()
predict_sent = mnb.predict_proba(sent_tfidf)
print("Vectorized Term Frequency- Inverse Document Frequency(TF-IDF)") 
print(sent_tfidf)

print()
maximum = 0
index   = 0

print("PROBABILITIES OF THE SENTENCE WITH GIVEN CLASSES:")
for i in range(len(train.target_names)):
    print(train.target_names[i],":",predict_sent[0][i])
    if(predict_sent[0][i] > maximum):
        maximum = predict_sent[0][i]
        index = i

print()
print("PREDICTED CLASS:",train.target_names[index])
print()




