'''
#######################################
각각의 Scaler의 특성과 정의 정리해놓기!
#######################################
원본 데이터는 데이터 고유의 특성과 분포를 가지고 있다.
이를 그대로 사용하게 되면 학습이 느리거나 문제가 발생하는 경우가 종종 발생
Scaler를 이용하여 동일하게 일정 범위로 스케일링하는 것이 필요

#Standard Scaler
기본 스케일. 평균과 표준편차 사용
데이터의 최소 최대를 모를때 사용
이상치가 있다면 평균과 표준편차에 영향을 미치기 때문에 데이터의 확산이 달라지게 됨
#Robust Scaler
중앙값과 IQR(interquartile range) 사용. 아웃라이어의 영향을 최소화.
모든 피처가 같은 크기를 갖는 다는 점이 standard와 유사
standard에 비해 이상치의 영향이 적어짐
#MaxAbs Scaler
최대절대값과 0이 각각 1,0이 되도록 스케일링. 절대값이 0~1사이에 매핑되도록 한다. / 큰 이상치에 민감할 수 있음
#MinMax Scaler
최대/최소값이 각각 1,0이 되도록 스케일링
'''

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score

# 1) 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

# print(np.min(x), np.max(x))  # 최솟값, 최댓값 출력  0.0 711.0
# x = x/711.  # 부동소수점으로 나눈다! 위아래 같음
# x = x/np.max(x) # 전처리   #boston은 컬럼이 13개 이므로 각각 특성마다 최소, 최댓값이 다를것임!

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)  # shuffle은 디폴트가 True

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MaxAbsScaler()

scaler.fit(x_train)  # 어느 비율로 나눌지
x_train = scaler.transform(x_train)  # 변환한 값을 다시 x_train에 넣어주자!
x_test = scaler.transform(x_test)  # x_train에 맞는 비율로 들어가있움

print(x.shape)  # (506, 13)
print(y.shape)  # (506,)

# 2) 모델구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=13))
model.add(Dense(10, activation='relu'))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

# model.summary()
'''
(506, 13)
(506,)
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_8 (Dense)             (None, 10)                140       

 dense_9 (Dense)             (None, 10)                110       

 dense_10 (Dense)            (None, 10)                110       

 dense_11 (Dense)            (None, 10)                110       

 dense_12 (Dense)            (None, 1)                 11        

=================================================================
Total params: 481
Trainable params: 481
Non-trainable params: 0
'''

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

# 4) 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

'''
#MinMaxScaler
loss:  16.505163192749023
r2스코어 :  0.8002210076302227

#StandardScaler
loss:  14.808168411254883
r2스코어 :  0.8207614737759786

#RobustScaler
loss:  9.629612922668457
r2스코어 :  0.8834428551414365

#MaxAbsScaler
loss:  15.176043510437012
r2스코어 :  0.8163087006332043
'''