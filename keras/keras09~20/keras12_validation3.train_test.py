from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.625, shuffle=False, random_state=66)  #16개니까 16으로 나눔.. 6.25인데 train 10개 *10
x_train, x_val, y_train, y_val = train_test_split(x_test,y_test, train_size=0.5, random_state=66)  #즉 train 10개 test(validation= 3:3)
print(x_train) # [14 16 15]
print(x_test) # [11 12 13 14 15 16]
print(x_val) # [11 12 13]

#2. 모델구성
model = Sequential()
model.add(Dense(21,input_dim=1))
model.add(Dense(17))
model.add(Dense(10))
model.add(Dense(6))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1, validation_data=(x_val, y_val))

#4. 평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict([13])
print("13의 예측값: ", y_predict)

'''
loss:  0.0016220324905589223
13의 예측값:  [[13.03556]]
'''