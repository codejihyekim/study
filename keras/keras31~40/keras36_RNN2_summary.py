import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# RNN은 3차원에서 2차원으로 빠진다/ step 명시/ RNN = 뒤로 가면 갈수록 기억 소실(=뒤로 갈수록 그 데이터를 많이 반영함)

# 1. 데이터
x = np.array([[1, 2, 3],
              [2, 3, 4],
              [3, 4, 5],
              [4, 5, 6]])  # timesteps: x와 y를 자르는 부분(크기) = 3
y = np.array([4, 5, 6, 7])  # feature : 1 (훈련시키는 크기)

print(x.shape, y.shape)  # (4, 3) (4,)

# input_shape = (batch_size, timesteps, feature)
# input_shape = (행, 열, 몇개씩 자르는지!!)  = 3차원(내용과 순서 안바뀜 = reshape)
x = x.reshape(4, 3, 1)  # 4행 3열 1개씩 자르겠다 ([[1],[2],[3]],...)  ---> 여기서 2차원을 3차원으로 바꿔줌

# 2. 모델구성
model = Sequential()
model.add(SimpleRNN(10, activation='linear', input_shape=(3, 1)))  # input_shape할 때는 행은 무시
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()
# (노드*노드) + (노드*피처) + (노드*바이어스)
# ((input+bias) * output) + (output*output)
# ((1+1)*10) + (10*10) = 120
'''
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 simple_rnn_1 (SimpleRNN)    (None, 10)                120       

 dense_3 (Dense)             (None, 10)                110       

 dense_4 (Dense)             (None, 1)                 11        

=================================================================
Total params: 241
Trainable params: 241
Non-trainable params: 0
_________________________________________________________________
'''