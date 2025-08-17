**1D-CNN**

import pandas as pd
import numpy as np
#reading dataset
df=pd.read_csv('/content/drive/My Drive/Hate Speech Detection/Project Codes/naive inp.csv')

import keras
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100
#tokenizing data
tokenizer =keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['tweet'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#sequence creation
X = tokenizer.texts_to_sequences(df['tweet'].values)
X =keras.preprocessing.sequence. pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

#one hot
Y = pd.get_dummies(df['class']).values
print('Shape of label tensor:', Y.shape)

from sklearn.model_selection import train_test_split
#splitting data to test and train
X_train, X_test, Y_train, Y_test =train_test_split(X,Y, test_size = 0.30, random_state = 4)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

from keras.models import Sequential 
from keras.layers import Dense, Flatten
from keras.layers import Embedding,SpatialDropout1D,LSTM,Bidirectional
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.callbacks import  EarlyStopping
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
#model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
model.add(Conv1D(128, 5, activation='relu')) 
model.add(Conv1D(128, 5, activation='relu')) 
model.add(MaxPooling1D()) 
#model.add(layers.GlobalMaxPooling1D()) 

model.add(Flatten())
model.add(Dense(50, activation='relu'))
#model.add(50, activation='relu')
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 10
batch_size = 400

#training
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)#,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]

one_y = Y_train.argmax(-1)

#predicting and Model Validation on train data set
pred = model.predict_classes(X_train, verbose=1)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(confusion_matrix(one_y,pred))
print(classification_report(one_y,pred))