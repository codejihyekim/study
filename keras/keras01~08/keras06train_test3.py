from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터 
x = np.array(range(100))
y = np.array(range(1, 101))

# train : test = 7 : 3

'''
x,y를 train과 test로 원하는 비율로 나누고 값들을 랜덤하게 뽑아주는 작업까지 모두 한번에 
from sklearn.model_selection import train_test_split 이 기능을 가져와서 쓸 수 있다. 
'''
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, shuffle=True, random_state=66) 
# random_state는 일정한 값이 나오게 해준다. 
# train_size를 0.7로 설정함으로 7:3 비율로 나눠진다.

print(x_test)
print(y_test)
'''
[ 8 93  4  5 52 41  0 73 88 68 25 18 26 29 66 50 80 45 38 58 49 85 94 87
 15  3 14 33 23 24]
[ 9 94  5  6 53 42  1 74 89 69 26 19 27 30 67 51 81 46 39 59 50 86 95 88
 16  4 15 34 24 25]
훈련을 반복해도 동일한 값이 나와야 제대로 된 훈련이 가능하기 때문에
이게 없으면 한번 다시 돌릴때마다 x_train~y_test 값이 계속 바뀐다.  
'''

#2. 모델링
model = Sequential()
model.add(Dense(3, input_dim=1))  
model.add(Dense(30)) 
model.add(Dense(7))
model.add(Dense(50))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train ,y_train , epochs=100, batch_size=1)

#04. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict([100]) 
print('101의 예측값: ', y_predict)

'''
loss:  0.0008493794593960047
101의 예측값:  [[101.060524]]
'''