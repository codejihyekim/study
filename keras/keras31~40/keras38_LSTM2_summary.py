import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

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
model.add(LSTM(10, activation='linear', input_shape=(3, 1)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()
## activation = 'tanh'이 디폴트
# 'tanh'=  sigmoid fuction을 보완하고자 나온 함수이다. 입력신호를 (−1,1) 사이의 값으로 normalization 해준다.
# 거의 모든 방면에서 sigmoid보다 성능이 좋다. / 중앙값 0
# sigmoid = 0~1사이 / 중앙값 0.5
# LSTM :  RNN의 주요 모델 중 하나로, 장기 의존성 문제를 해결할 수 있음
# 직전 데이터뿐만 아니라, 좀 더 거시적으로 과거 데이터를 고려하여 미래의 데이터를 예측하기 위함

# (((1+1)*10) + (10*10))*4 = 480 (3개의 gate와 1개의 state이 있기 때문에 4를 곱해준다.)
'''
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm_1 (LSTM)               (None, 10)                480       

 dense_3 (Dense)             (None, 10)                110       

 dense_4 (Dense)             (None, 1)                 11        

=================================================================
Total params: 601
Trainable params: 601
Non-trainable params: 0
_________________________________________________________________
'''