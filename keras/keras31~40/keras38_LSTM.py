import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,LSTM

#1. 데이터
x = np.array([[1,2,3],
            [2,3,4],
            [3,4,5],
            [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape,y.shape)  # (4, 3) (4,)

# input_shape = (batch_size, timesteps, feature)
# input_shape = (행, 열, 몇개씩 자르는지!!)  = 3차원(내용과 순서 안바뀜 = reshape)
x = x.reshape(4, 3, 1) # 4행 3열 1개씩 자르겠다 ([[1],[2],[3]],...)  ---> 여기서 2차원을 3차원으로 바꿔줌

#2. 모델구성
model = Sequential()
model.add(LSTM(80, activation = 'linear', input_shape = (3,1)))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer= 'adam')
model.fit(x,y,epochs=100)

#4. 평가, 예측
model.evaluate(x,y)
result = model.predict([[[5],[6],[7]]])
print(result)
# [[8.430655]]