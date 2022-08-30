import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional, Conv1D, Flatten

# 1. 데이터
x = np.array([[1, 2, 3],
              [2, 3, 4],
              [3, 4, 5],
              [4, 5, 6]])
y = np.array([4, 5, 6, 7])
# print(x.shape, y.shape) # (4, 3) (4,) # 2차원

x = x.reshape(4, 3, 1)  # 3차원으로 변경

# 2. 모델구성
model = Sequential()
model.add(Conv1D(10, 2, input_shape=(3, 1)))  # 10 -> filter, 2 -> kernel size 2씩 묶어주겠다.
model.add(Dense(60, activation='relu'))
model.add(Flatten())  # 1차원으로 변경
model.add(Dense(1))
model.summary()
'''
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 2, 10)             30        

 dense_2 (Dense)             (None, 2, 60)             660       

 flatten (Flatten)           (None, 120)               0         

 dense_3 (Dense)             (None, 1)                 121       

=================================================================
Total params: 811
Trainable params: 811
Non-trainable params: 0
_________________________________________________________________
'''