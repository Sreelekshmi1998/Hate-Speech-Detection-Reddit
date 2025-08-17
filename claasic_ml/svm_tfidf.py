import numpy as np
import pandas as pd
import re
import nltk
dataset = pd.read_csv("/content/drive/My Drive/Hate Speech Detection/Project Codes/naive inp.csv")
def removeSpaces(dataset):
  ret = []
  for data in dataset:
    ret.append(data.strip())

  return ret

X = dataset.drop('class',axis=1)
x = list(X["tweet"])
X = removeSpaces(x)
y = list(dataset['class'])
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
X = count.fit_transform(X)

tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()
y_ = np.array(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_, test_size=0.3, random_state=4)
from sklearn.svm import SVC
text_classifier = SVC(kernel='linear',probability=True)
text_classifier.fit(X_train, y_train)
predictions = text_classifier.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))
import os
os.chdir("/content/drive/My Drive/Hate Speech Detection/Project Codes/Voting_input/")

from joblib import dump, load
#to save model
dump(text_classifier, 'tfidf.joblib')