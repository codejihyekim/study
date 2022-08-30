from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Bidirectional,Conv1D,Flatten, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import time

#1 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

#print(x.shape, y.shape) # (442, 10) (442,)

x = x.reshape(442,10,1)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, shuffle=True, random_state=66)
scaler = RobustScaler()

print(x_train.shape, x_test.shape) # (309, 10, 1) (133, 10, 1)

x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1)).reshape(x_train.shape)
x_test = scaler.transform(x_test.reshape(len(x_test), -1)).reshape(x_test.shape)

# 2.모델링
model = Sequential()
model.add(Conv1D(80,2,input_shape = (10,1)))
model.add(Flatten())
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1))

'''model = Sequential()
model.add(LSTM(80, input_shape = (10,1))) 
model.add(Dense(45))
model.add(Dense(1))'''

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto',restore_best_weights=True)
start = time.time()
model.fit(x_train, y_train, epochs=1000,validation_split=0.2 ,callbacks=[es], batch_size=300)
end = time.time() - start
print('걸린시간: ', round(end, 3))

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict)
print('r2: ', r2)
'''
-Conv1D-
loss:  3062.460205078125
r2:  0.5084625614926759

-LSTM-
loss:  6228.91455078125
r2:  0.00023363276579713155
'''