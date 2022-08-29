import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# keras.io 참고!!

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
# model.add(SimpleRNN(10, input_shape = (3,1))) # input_shape할 때는 행은 무시
# model.add(SimpleRNN(units = 10, input_shape = (3,1))) # units: Positive integer(노드/output)
model.add(SimpleRNN(10, input_length=3, input_dim=1))  # 위에 3개가 다 같은 의미!
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()
'''
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 simple_rnn_2 (SimpleRNN)    (None, 10)                120       

 dense_5 (Dense)             (None, 10)                110       

 dense_6 (Dense)             (None, 1)                 11        

=================================================================
Total params: 241
Trainable params: 241
Non-trainable params: 0
_________________________________________________________________
'''