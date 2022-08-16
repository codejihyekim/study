from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

#2. 모델구성
model = Sequential()
model.add(Dense(44, input_dim=1))
model.add(Dense(11))
model.add(Dense(33))
model.add(Dense(44))
model.add(Dense(44))
model.add(Dense(44))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
start = time.time()
model.fit(x, y, epochs=1000, batch_size=1, verbose=2)
end = time.time() - start
print("걸린시간 : ", end)

# 0: 걸린시간: 10.483530044555664
# 1: 걸린시간: 26.16771697998047
# 2: 걸린시간: 20.927370309829712

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x)

r2 = r2_score(y, y_predict)
print('r2스코어 : ', r2)

'''
loss :  0.4224814474582672
r2스코어 :  0.7887592935753205
'''