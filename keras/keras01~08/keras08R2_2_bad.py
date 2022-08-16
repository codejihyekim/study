# R2를 음수가 아닌 0.5 이하로 만들것 
# 데이터 건들지 않기 
# 레이어는 인풋 아웃풋 포함 6개 이상 
# epoch는 100이상 
#  train 70%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터 
x = np.array(range(100))
y = np.array(range(1, 101))

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(300))
model.add(Dense(350))
model.add(Dense(444))
model.add(Dense(10))
model.add(Dense(444))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(250))
model.add(Dense(150))
model.add(Dense(444))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
# loss : 0.16291183233261108
y_predict = model.predict([x_test])

r2 = r2_score(y_test,y_predict)
print('r2스코어 : ', r2)
# r2스코어 :  0.9998136835021125 1에 가까울 수록 연관성이 높다 