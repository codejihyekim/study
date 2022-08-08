import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#01. 데이터 
x = np.array([[1,2,3,4,5,6,7,8,9,10], 
            [1,1.1,1.2,1.3,1.4,1.5,
              1.6,1.5,1.4,1.3]])
y = np.array([11,12,13,14,15,16,17,18,19,20])

print(x.shape) #(2,10)

x = x.T
print(x)
'''
[[ 1.   1. ]
 [ 2.   1.1]
 [ 3.   1.2]
 [ 4.   1.3]
 [ 5.   1.4]
 [ 6.   1.5]
 [ 7.   1.6]
 [ 8.   1.5]
 [ 9.   1.4]
 [10.   1.3]]

print(x.shape) #(10,2)
print(y.shape) #(10,)'''

'''x = np.transpose(x)
print(x.shape) #(10,2)
x = x.reshape(10, 2)
print(x)'''

# 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2))  
model.add(Dense(15)) 
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x ,y , epochs=30, batch_size=1)


#04. 평가, 예측
loss = model.evaluate(x, y)
print('loss: ', loss)
y_predict = model.predict([[10, 1.3]])
print('[10, 1.3]의 예측값: ', y_predict)

'''
loss:  0.5136356353759766
[10, 1.3]의 예측값:  [[18.68774]]
'''
