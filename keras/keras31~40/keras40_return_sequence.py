import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# LSTM을 두개 엮으면 성능이 더 bad

# 1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9],
              [8, 9, 10], [9, 10, 11], [10, 11, 12], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

print(x.shape, y.shape)  # (13, 3) (13,)

# input_shape = (batch_size, timesteps, feature)
x = x.reshape(13, 3, 1)

# 2. 모델구성
model = Sequential()
model.add(LSTM(10, return_sequences=True,
               input_shape=(3, 1)))  # (N,3,1) -> (N,3,10)  # return_sequences=True를 사용하면 3차원으로 change
model.add(LSTM(20, return_sequences=True))  # (None, 3, 20)                                 # return_sequences = false가 디폴트
model.add(LSTM(30, return_sequences=True))  # (None, 3, 30)
model.add(LSTM(40, return_sequences=True))  # (None, 3, 40)
model.add(LSTM(50, return_sequences=False))  # (None, 50)
model.add(Dense(1))  # dense은 원래 위에 차원 그대로 받아준다.

# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')

es = EarlyStopping(monitor='loss', patience=100, mode='min', verbose=1, restore_best_weights=True)

start = time.time()
model.fit(x, y, epochs=10000, callbacks=[es], batch_size=128)
end = time.time() - start
print("걸린시간 : ", round(end, 3))

# 4. 평가, 예측
model.evaluate(x, y)
y_pred = np.array([50, 60, 70]).reshape(1, 3, 1)
x_predict = model.predict(y_pred)

print(x_predict)
'''
Epoch 2132: early stopping
1/1 [==============================] - 3s 3s/step - loss: 0.0959
[[70.07639]]
'''