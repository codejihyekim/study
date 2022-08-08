# 데이터를 train과 test로 나눠주는 이유 
# 모델 학습에 사용했던 훈련데이터를 잘 맞추는 모델이 아니라,
# 학습에 사용하지 않은 테스트 데이터를 얼마나 잘 맞추는지가 중요하다. 
# train 데이터는 우리가 학습을 할 때 사용할 데이터 
# test 데이터는 우리가 학습한 모델의 성능을 텡스트하는 데이터이다. 
# 이 때 test 데이터는 한번도 공개된 적이 없는 데이터여야 한다. 
# 즉 과적합을 피하고 편향을 제거한 데이터로 모델 성능을 평가하기 위해서이다. 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터 
x_train = np.array([1,2,3,4,5,6,7]) # 훈련
x_test = np.array([8,9,10]) #평가
y_train = np.array([1,2,3,4,5,6,7])
y_test = np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([11])
print('11의 예측값 : ', result)

'''
loss :  0.09604617208242416
11의 예측값 :  [[10.548395]]
'''