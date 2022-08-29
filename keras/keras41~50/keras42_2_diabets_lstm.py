from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score

datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, shuffle=True,random_state=66)

print(x_train.shape, x_test.shape) # (309, 10) (133, 10)

scaler = RobustScaler()

x_train = scaler.fit_transform(x_train).reshape(309,10,1)
x_test = scaler.transform(x_test).reshape(133,10,1)

#2 모델링
model = Sequential()
model.add(LSTM(80, input_shape = (10,1)))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dropout(0.5))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)
model.fit(x_train, y_train, epochs = 1000, validation_split=0.2, callbacks=[es], batch_size = 128)

#4 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2: ', r2)
'''
-LSTM-
loss:  4868.5869140625
r2:  0.21857185857994144
'''