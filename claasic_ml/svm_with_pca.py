from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
dataset=pd.read_csv("/content/drive/My Drive/Hate Speech Detection/Project Codes/unigramSvmInput/neutral.csv")
X = dataset.drop('lab_el', axis=1)
y = dataset['lab_el']
print(X.shape)
print(y.shape)
classifier = SVC()
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.3,random_state=4)
pca = PCA()
X_transformed = pca.fit_transform(X_train)
classifier.fit(X_transformed, y_train)
newdata_transformed = pca.transform(X_test)
pred_labels = classifier.predict(newdata_transformed)
print("pred_labels\n",pred_labels)
print(classification_report(y_test,pred_labels))
print("confusion matrix")
print(confusion_matrix(y_test,pred_labels))
import os
os.chdir("/content/drive/My Drive/Hate Speech Detection/Project Codes/Voting_input/")

from joblib import dump, load
#to save model
dump(classifier, 'svm_pca_neutral.joblib')