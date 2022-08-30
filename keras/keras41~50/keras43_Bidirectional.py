import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional

# 소문자 - 함수 / 대문자 - 클래스
# LSTM의 단방향의 한계(?)를 개선한 모델! = Bidrectional(양방향)- 이거는 방향성만 제시해주는 것이므로 무엇을 쓸건지 정의해줘야함!

#1. 데이터
x = np.array([[1,2,3],
            [2,3,4],
            [3,4,5],
            [4,5,6]])
y = np.array([4,5,6,7])
print(x.shape, y.shape) # (4, 3) (4,)

x = x.reshape(4,3,1)

#2 모델 구성
model = Sequential()
model.add(Bidirectional(SimpleRNN(10), input_shape=(3,1))) # SimpleRNN을 양방향으로 돌리겠다.
model.add(Dense(60, activation='relu'))
model.add(Dense(1))
model.summary()
'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 bidirectional (Bidirectiona  (None, 20)               240       
 l)                                                              
                                                                 
 dense (Dense)               (None, 60)                1260      
                                                                 
 dense_1 (Dense)             (None, 1)                 61        
                                                                 
=================================================================
Total params: 1,561
Trainable params: 1,561
Non-trainable params: 0
_________________________________________________________________
'''