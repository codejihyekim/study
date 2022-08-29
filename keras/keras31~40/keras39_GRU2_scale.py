import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU
import time

# 80을 만들어랏!
# 성능이 유사할 경우 - fit에 time 걸어서 속도 확인

# 1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9],
              [8, 9, 10], [9, 10, 11], [10, 11, 12], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

print(x.shape, y.shape)  # (13, 3) (13,)

# input_shape = (batch_size, timesteps, feature)
x = x.reshape(13, 3, 1)

# 2. 모델구성
model = Sequential()
model.add(LSTM(170, activation='relu', input_shape=(3, 1)))  # input_shape할 때는 행은 무시
model.add(Dense(90, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')

start = time.time()
model.fit(x, y, epochs=100)
end = time.time() - start
print("걸린시간 : ", round(end, 3))

# 4. 평가, 예측
model.evaluate(x, y)
y_pred = np.array([50, 60, 70]).reshape(1, 3, 1)
x_predict = model.predict(y_pred)
# x_predict = model.predict([[ [50], [60],[70] ]])

print(x_predict)

'''
[[84.82451]] - LSTM 걸린시간  6.544
[[85.05955]] - GRU 걸린시간 4.9
'''