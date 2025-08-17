import pandas as pd
df=pd.read_csv('/content/drive/My Drive/Hate Speech Detection/Project Codes/unigramSvmInput/hate.csv')
print(df.shape)


X=df.drop('lab_el',axis=1)
y=df['lab_el']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=4)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_predict=rfc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,rfc_predict))
print(classification_report(y_test,rfc_predict))
print(accuracy_score(y_test,rfc_predict))

import os
os.chdir("/content/drive/My Drive/Hate Speech Detection/Project Codes/Voting_input/")
from joblib import dump,load
dump(rfc,'rf_hate.joblib')