from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#1. 데이터 
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,9,8,12,13,17,12,14,21,14,11,19,23,25])

x_train, x_test, y_train, y_test = train_test_split(x,y,
         train_size=0.7, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(170))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x_train, y_train, epochs=150, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)   
print('loss : ', loss)
#loss :  8.73421573638916

y_predict = model.predict([x_test])

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2) # 회귀 모델의 성능에 대한 평가 지표
# R2 score는 0~1 사이의 범위를 가지고 1에 가까울수록 해당 선형 회귀 모델이 해당 데이터에 대한 높은 연관성을 가지고 있다 
#r2스코어 :  0.38587545708604865

