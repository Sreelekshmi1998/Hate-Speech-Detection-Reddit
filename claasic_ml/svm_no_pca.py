'''!apt-get -qq install request
!apt-get -qq install pandas'''
import pandas as pd
dataset = pd.read_csv('/content/drive/My Drive/Hate Speech Detection/Project Codes/unigramSvmInput/offensive.csv')
#dataset.drop(dataset.columns[dataset.columns.str_contains('unnamed',case = False)],axis = 1, inplace = True)
#print(dataset)
x = dataset.drop('lab_el', axis=1)
y = dataset['lab_el']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .20)
from sklearn.svm import SVC
svclassifier=SVC(probability=True)
svclassifier.fit(x_train, y_train)
y_pred = svclassifier.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_pred,y_test))
print("confusion matrix")
print(confusion_matrix(y_pred,y_test))
import os
os.chdir("/content/drive/My Drive/Hate Speech Detection/Project Codes/Voting_input/")

from joblib import dump, load
#to save model
dump(svclassifier, 'svm_offensive.joblib')