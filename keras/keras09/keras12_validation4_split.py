from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8125, shuffle=True, random_state=66)
print(x_train) # [16  9  1 10  3 15  6 12 11  4  8 13  5]
print(x_test) # [ 7  2 14]

#2. 모델구성
model = Sequential()
model.add(Dense(21,input_dim=1))
model.add(Dense(17))
model.add(Dense(10))
model.add(Dense(6))
model.add(Dense(1))

#3. 컴파일, 훈련
#train_test_split로 나누시오 (10:3:3)
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.3)   #validation_data=(x_val, y_val))

#4. 평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict([18])
print("18의 예측값: ", y_predict)

'''
loss:  1.288450791371576e-12
18의 예측값:  [[17.999998]]
'''