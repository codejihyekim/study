import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

#1) 데이터
datasets = fetch_covtype()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)   # (581012, 54) (581012,)  ----> (다중분류)
print(np.unique(y))    # [1 2 3 4 5 6 7]  --->다중분류임을 알 수 있음
''' 
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)   #One Hot Encoding
print(y)
print(y.shape) # (581012, 8) 원핫인코딩     #categorical은 앞에 0부터 시작 그래서 8로 나옴
'''
'''
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)  #if sparse = True면 metrics로 출력, False면 array로 출력
y = ohe.fit_transform(y.reshape(-1,1))  #1부터 시작 ~ -1즉 배열 끝까지 출력!...
print(y.shape)  # (581012, 7)
'''
#2)
# pandas
import pandas as pd   # get_dummies) 결측값이 사라져서 수가 줄어듦 (581012, 7) 출력됨 / dummy_na = True) 결측값도 인코딩 처리되서 (581012, 8) 출력됨
y = pd.get_dummies(y)   # pandas는 수치로 보여줌!  그 외 pandas와 sklearn은 비슷!!
print(y.shape)    #(drop_first=True)  # 열을 n-1개 생성 / (581012, 7)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2) 모델링
model = Sequential()
model.add(Dense(10, activation= 'linear', input_dim = 54))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(7, activation = 'softmax'))

#3) 캄파일, 훈련
model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs = 100, batch_size=500, validation_split= 0.2, callbacks=[es])   # 데이터가 크기 때문에 batch_size를 크게 해줌!

#4) 평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)
results= model.predict(x_test[:11])
print(y_test[:11])
print('results: ', results)
'''
loss:  [0.6555075645446777, 0.7107734084129333]
        1  2  3  4  5  6  7
257457  1  0  0  0  0  0  0
15362   0  1  0  0  0  0  0
455621  1  0  0  0  0  0  0
26237   0  1  0  0  0  0  0
530518  0  1  0  0  0  0  0
2113    0  1  0  0  0  0  0
81459   0  1  0  0  0  0  0
348766  0  0  1  0  0  0  0
552259  1  0  0  0  0  0  0
239314  0  0  0  0  0  1  0
201753  1  0  0  0  0  0  0
results:  [[6.7901903e-01 2.9189140e-01 5.5563874e-08 5.4347798e-12 1.8603781e-03
  4.4363560e-08 2.7229086e-02]
 [9.4816193e-02 8.9748293e-01 5.8590380e-05 2.9844682e-06 7.0146178e-03
  5.4313592e-04 8.1422848e-05]
 [7.6161885e-01 2.2083330e-01 8.5698002e-07 4.8009270e-11 2.7314273e-03
  8.7866883e-06 1.4806836e-02]
 [7.2945356e-02 9.1120064e-01 2.7297472e-04 1.1900140e-03 1.3586635e-02
  5.1613728e-04 2.8819480e-04]
 [4.0848669e-01 5.7776278e-01 1.6780310e-05 1.8654230e-10 3.9314530e-03
  3.2321517e-05 9.7699584e-03]
 [1.9958311e-01 7.6432979e-01 1.6797023e-04 5.6177552e-11 3.5794914e-02
  6.5636901e-05 5.8625214e-05]
 [1.5982142e-01 8.3746809e-01 2.2220171e-05 1.1338312e-06 1.1914660e-03
  2.8738259e-05 1.4668832e-03]
 [1.6404176e-03 1.4589640e-01 6.8833691e-01 5.5782334e-04 7.3750414e-02
  8.9817159e-02 8.5255760e-07]
 [1.9505687e-01 7.8601289e-01 3.0896728e-04 2.3913318e-08 1.7037209e-02
  7.7287800e-04 8.1111479e-04]
 [3.9727631e-08 6.4071675e-05 7.5361454e-01 1.0462224e-02 2.4796833e-05
  2.3583427e-01 2.7093334e-12]
 [6.6234094e-01 3.3628249e-01 2.8801338e-07 1.9925240e-12 7.8688195e-04
  7.0398369e-07 5.8870920e-04]]
'''