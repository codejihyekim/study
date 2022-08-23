import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

'''
activation(relu) : layer에서 다음 layer로 넘어갈 때 음수는 0으로 해주고 양수만 남겨주는 함수! if 음수값이 사라지면 성능이 조아진다!! 
'''

# 1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

# 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

model.summary()  # y=wx+b 모든 연산에는 바이어스까지 해줘야함... ex) 위와 같은 모델일 경우 1*5=5(x) / 1+1(바이어스)*5=10(o)
'''
Model: "sequential_17"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_97 (Dense)            (None, 5)                 10        

 dense_98 (Dense)            (None, 3)                 18        

 dense_99 (Dense)            (None, 4)                 16        

 dense_100 (Dense)           (None, 2)                 10        

 dense_101 (Dense)           (None, 1)                 3         

=================================================================
Total params: 57
Trainable params: 57
Non-trainable params: 0
'''
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=30, batch_size=1)
# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([4])
print('4의 예측값 : ', result)

'''
loss :  2.4310495853424072
4의 예측값 :  [[0.779406]]
'''