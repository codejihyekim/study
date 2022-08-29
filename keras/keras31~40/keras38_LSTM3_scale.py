import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,Dropout,LSTM

# 80을 만들어랏!

# LSTM
# 1) forget gate layer(sigmoid): 어느 부분이 삭제되어야하는지 제어하는 역할 / 1에 가까울수록 정보 반영 많이함.. 0은 그 반대
# 2) input gate layer : 장기 상태의 어느 부분이 기억되어야 하는지 제어하는 역할 / 새로운 정보가 cell state에 저장이 될지를 결정하는 게이트
# 3) update cell state : Input, Forget, Output 게이트의 정보 반영
# 4) output gate layer : 업데이트 된 cell state를 tanh을 통해 -1~1사이로 출력해줌(출력 값 결정단계)

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],
              [8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape,y.shape)  # (13, 3) (13,)

# input_shape = (batch_size, timesteps, feature)
x = x.reshape(13, 3, 1)

#2. 모델구성
model = Sequential()
model.add(LSTM(170, activation= 'relu' ,input_shape = (3,1))) # input_shape할 때는 행은 무시
model.add(Dense(90, activation = 'relu'))
model.add(Dense(60, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer= 'adam')
model.fit(x,y,epochs=100)

#4. 평가, 예측
model.evaluate(x,y)
y_pred = np.array([50,60,70]).reshape(1,3,1)
x_predict = model.predict(y_pred)
#x_predict = model.predict([[ [50], [60],[70] ]])

print(x_predict)
# [[95.74549]]