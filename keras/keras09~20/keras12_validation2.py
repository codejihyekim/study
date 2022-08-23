from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))
x_train = x[:11]
y_train = y[:11]
x_test = x[11:14]
y_test = y[11:14]
x_val = x[14:17]
y_val = y[14:17]

'''
#1. 데이터
x_train = np.array(range(1,10))
y_train = np.array(range(1,10))
x_test = np.array([11,12,13])
y_test = np.array([11,12,13])
x_val = np.array([14,15,16])
y_val = np.array([14,15,16])
'''

#2. 모델구성
model = Sequential()
model.add(Dense(17,input_dim=1))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val)) # loss보다 val_loss를 더 신뢰

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict([10])
print('10의 예측값: ',y_predict)

'''
loss:  2.5162686170809856e-11
10의 예측값:  [[10.000004]]
'''