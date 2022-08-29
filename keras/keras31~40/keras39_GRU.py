import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

# 1. 데이터
x = np.array([[1, 2, 3],
              [2, 3, 4],
              [3, 4, 5],
              [4, 5, 6]])
y = np.array([4, 5, 6, 7])

print(x.shape, y.shape)  # (4, 3) (4,)

# input_shape = (batch_size, timesteps, feature)
# input_shape = (행, 열, 몇개씩 자르는지!!)  = 3차원(내용과 순서 안바뀜 = reshape)
x = x.reshape(4, 3, 1)  # 4행 3열 1개씩 자르겠다 ([[1],[2],[3]],...)  ---> 여기서 2차원을 3차원으로 바꿔줌

# 2. 모델구성
model = Sequential()
model.add(GRU(10, activation='linear', input_shape=(3, 1)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

'''
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 gru (GRU)                   (None, 10)                390       

 dense_6 (Dense)             (None, 10)                110       

 dense_7 (Dense)             (None, 1)                 11        

=================================================================
Total params: 511
Trainable params: 511
Non-trainable params: 0
_________________________________________________________________
'''