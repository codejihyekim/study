# 3등분 (train, test, validation(검증))
# train = 컴퓨터 / test = 유저

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x_train = np.array(range(1,11))
y_train = np.array(range(1,11))
x_test = np.array([11,12,13])
y_test = np.array([11,12,13])
x_val = np.array([14,15,16])
y_val = np.array([14,15,16])

#2. 모델구성
model = Sequential()
model.add(Dense(10,input_dim=1))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val)) # loss보다 val_loss를 더 신뢰

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict([17])
print('17의 예측값: ',y_predict)

'''
17 맞추기 
loss:  0.008508831262588501
17의 예측값:  [[16.82094]]
'''