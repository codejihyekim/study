import numpy as np
from tensorflow.keras.models import Sequential, Model  # Model = 함수형 모델
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101)])
y = np.array([range(701, 801)])
print(x.shape, y.shape)  # (3,100)  (1,100)
x = np.transpose(x)
y = np.transpose(y)
print(x.shape, y.shape)  # (100, 3) (100, 1)

'''
x = x.reshape(1,10,10,3)
print(x.shape, y.shape)    # (1,10, 10, 3) (10, 1)   #이미지 데이터 = 기본 (4차원), 1장의 이미지가 가로 10 세로10 3장
                           # if reshape 할 때 갯수 맞춰줘야함!! 1*10*10*3=300
'''

# 2. 모델구성
model = Sequential()
# model.add(Dense(10, input_dim=3))   # (100,3)  -> (N,3)      #이렇게 했더니 다차원에서 문제가 있음! 그래서 대신 input_shape를 해줌!
model.add(Dense(10, input_shape=(
3,)))  # 2차원에서는 x기 (100,3)이므로 3으로 해줌!    # if x.reshape(1,10,10,3) 이거라면 맨 앞이 행이 된다! 여기서 행(맨 앞에있는애)은 무시하고 나머지만 써준다! 즉 10,10,3을 써준다!
model.add(Dense(9))  # 2차원에서는 input_dim과 input_shape는 같지만 3,4..차원에서는 input_dim으로는 안되므로 input_shape를 대신 써주자!
model.add(Dense(8))  # input_dim=3  === input_shape=(3,)
model.add(Dense(1))
model.summary()
'''
Model: "sequential_18"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_102 (Dense)           (None, 10)                40        

 dense_103 (Dense)           (None, 9)                 99        

 dense_104 (Dense)           (None, 8)                 80        

 dense_105 (Dense)           (None, 1)                 9         

=================================================================
Total params: 228
Trainable params: 228
Non-trainable params: 0
'''