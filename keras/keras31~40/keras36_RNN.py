import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# RNN은 3차원에서 2차원으로 빠진다/ step 명시/ RNN = 뒤로 가면 갈수록 기억 소실(=뒤로 갈수록 그 데이터를 많이 반영함)

#1. 데이터
x = np.array([[1,2,3],
            [2,3,4],
            [3,4,5],
            [4,5,6]])   # timesteps: x와 y를 자르는 부분(크기) = 3
y = np.array([4,5,6,7]) # feature : 1 (훈련시키는 크기)

print(x.shape,y.shape)  # (4, 3) (4,)

# input_shape = (batch_size, timesteps, feature)
# input_shape = (행, 열, 몇개씩 자르는지!!)  = 3차원(내용과 순서 안바뀜 = reshape)
x = x.reshape(4, 3, 1) # 4행 3열 1개씩 자르겠다 ([[1],[2],[3]],...)  ---> 여기서 2차원을 3차원으로 바꿔줌

#2. 모델구성
model = Sequential()
model.add(SimpleRNN(80, activation = 'linear', input_shape = (3,1))) # input_shape할 때는 행은 무시
model.add(Dense(60, activation = 'relu'))
model.add(Dense(35, activation = 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer= 'adam') #loss 최적화
model.fit(x,y,epochs=100)  # batch_size는 자동으로 32

#4. 평가, 예측
model.evaluate(x,y)
result = model.predict([[[5],[6],[7]]])
print(result)
# [[8.354761]]