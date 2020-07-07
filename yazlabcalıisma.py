#Decission Tree
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('Iris.csv')
df.drop(["Id"],axis=1, inplace= True)
species = df.iloc[:,-1:].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,1:-1],species,test_size=0.80,random_state=42)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(x_train,y_train)
print("score",dtc.score(x_test, y_test))
result = dtc.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, result)
print(accuracy)

#%%
#RNN
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing import sequence


data = pd.read_csv('Iris.csv',engine='python')
data.drop(["Id"],axis=1, inplace= True)
s=1
ve=2
vi=3
data["Species"] = data["Species"].map({'Iris-setosa': s, 'Iris-versicolor': ve,'Iris-virginica':vi})
data=data.iloc[np.random.permutation(len(data))]
X=data.iloc[:,-1:].values
y=data.iloc[:,:-1].values

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,:-1].values, X, test_size=0.8, random_state=30) 
max_features=20000
maxlen=200
x_train=sequence.pad_sequences(x_train,maxlen=maxlen)
x_test=sequence.pad_sequences(x_test, maxlen=maxlen)
inputs=keras.Input(shape=(None,), dtype="int32")
x=layers.Embedding(max_features,128)(inputs)
x=layers.Bidirectional(layers.LSTM(64))(x)
outputs=layers.Dense(1,activation="sigmoid")(x)
model=keras.Model(inputs,outputs)
model.compile("adam","binary_crossentropy",metrics=["accuracy"])
model.fit(x_train,y_train,batch_size=32,epochs=10,validation_data=(x_test,y_test))
model.summary()
