from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터 정제해서 값 도출 
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성 layer와 parameter를 추가해서 deep 러닝으로 만들어본다.
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(21))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(18))
model.add(Dense(10)) # 중간 레이어들 = 히든 레이어 히든레이어 값 바꾸는 것 -> 하이퍼파라미터 튜닝이라고 한다. 
model.add(Dense(8))
model.add(Dense(1)) # 마지막 레이어 = 아웃풋레이어 


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=50, batch_size=1)

#4. 평가, 예측 
loss = model.evaluate(x,y)
print('loss : ', loss)
result = model.predict([4])
print('4의 예측값 : ', result)

'''
loss :  0.0005962526774965227
4의 예측값 :  [[3.9502285]]
'''

'''
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(21)) -> new
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(18)) -> new
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(1))

loss :  3.1221305107465014e-05
4의 예측값 :  [[3.989421]]
'''