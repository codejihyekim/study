from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target  # y: 0 or 1
#print(x.shape, y.shape)   # (569, 30) (569,)

#print(np.unique(y))   # [0 1]  : 이진분류

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

scaler = MinMaxScaler()

n = x_train.shape[0]
x_train_transe = scaler.fit_transform(x_train)
#print(x_train_transe.shape)   # (455, 30)
x_train = x_train_transe.reshape(n,2,5,3)
#print(x_train.shape)   # (455, 2, 5, 3)

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(m,2,5,3)

#2. 모델 구성

model = Sequential()
model.add(Conv2D(10, kernel_size=(2, 2), padding='same', input_shape=(2,5,3)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=30, mode='min', restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_split=0.2, callbacks=[es])

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("R2 : ",r2)
'''
loss :  0.06485974788665771
R2 :  0.7212575137288146
'''