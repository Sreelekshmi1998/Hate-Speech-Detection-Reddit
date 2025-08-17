from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

#load the data set
dataset=pd.read_csv('/content/drive/My Drive/Hate Speech Detection/Project Codes/unigramSvmInput/offensive.csv')

#load the data set

X=dataset.drop('lab_el',axis=1)
y=dataset['lab_el']
print(X.shape)
print(y.shape)


# initiate PCA and classifier
pca = PCA()
classifier = RandomForestClassifier()

X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.3,random_state=4)

# transform / fit 			[a x1+b x2+c x3……..]
X_transformed = pca.fit_transform(X_train)
classifier.fit(X_transformed, y_train)

# predict "new" data
# transform new data using already fitted pca
#X is projected on the first principal components previously extracted from a training set.
newdata_transformed = pca.transform(X_test)

# predict labels using the trained classifier
pred_labels = classifier.predict(newdata_transformed)
print(pred_labels)


from sklearn import metrics
print(metrics.accuracy_score(pred_labels,y_test))


from sklearn import model_selection
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_predict=rfc.predict(X_test)


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,rfc_predict))
print(confusion_matrix(y_test,rfc_predict))

import os
os.chdir('/content/drive/My Drive/Hate Speech Detection/Project Codes/Voting_input')
from joblib import dump,load
dump(classifier,'rf_offensive_pca.joblib')