from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score

datasets = load_diabetes() # 당뇨병 환자 데이터
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state=66)   #shuffle은 디폴트가 True

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = RobustScaler()
# scaler = MaxAbsScaler()

scaler.fit(x_train)  # 어느 비율로 나눌지
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) #x_train에 맞는 비율로 들어가있음

# print(x.shape)  # (442, 10)
# print(y.shape)    # (442,)

#2) 모델구성
model = Sequential()
model.add(Dense(100,input_dim=10))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))

#3) 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

#4) 평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

'''
# MinMaxScaler
loss:  3152.4013671875
r2스코어 :  0.494026682845105
# StandardScaler
loss:  3052.66943359375
r2스코어 :  0.5100340035389164
# RobustScaler
loss:  3118.02197265625
r2스코어 :  0.49954462098937336
# MaxAbsScaler
loss:  3130.363525390625
r2스코어 :  0.49756384952718036
'''