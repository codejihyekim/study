# 과제) 중위값과 평균값 비교 분석
'''
평균: 자료 모두 더한 후에 전체자료 갯수로 나눔
중위값: 자료를 크기 순으로 나열한 다음, 가장 중앙에 위치하는 값
중위값) 1) 자료의 개수가 홀수일 때 : (n+1)/2번째 관측값
2) 자료의 개수가 짝수일 때: n/2번째 관측값과 (n+1)/2번째 관측값의 평균
#mean 평균
#로그와 루트
#가장 잘 나온값 : 3.14784
'''

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test,y_pred)) #제곱근

#1)데이터
path = "/content/drive/MyDrive/bike/"      #.지금 현재 작업 공간   / ..이전

train = pd.read_csv(path + 'train.csv')
#print(train)   #(10886, 12)
test_file = pd.read_csv(path + 'test.csv')  #(6493,9)
#print(test)
submit_file = pd.read_csv(path + 'sampleSubmission.csv')
#print(submit)  #(6493, 2)

#print(type(train))         #<class 'pandas.core.frame.DataFrame'>
print(train.info())    #object는 string (문자) 일단 문자형으로 생각.. so 비교해주려면 문자를 숫자로 변환시켜줘야함
print(train.describe())   #pandas는 변수.describe /  sklearn에서는 변수.DESC
'''
             season       holiday    workingday       weather         temp         atemp      humidity     windspeed        casual    registered         count
count  10886.000000  10886.000000  10886.000000  10886.000000  10886.00000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000
mean       2.506614      0.028569      0.680875      1.418427     20.23086     23.655084     61.886460     12.799395     36.021955    155.552177    191.574132
std        1.116174      0.166599      0.466159      0.633839      7.79159      8.474601     19.245033      8.164537     49.960477    151.039033    181.144454
min        1.000000      0.000000      0.000000      1.000000      0.82000      0.760000      0.000000      0.000000      0.000000      0.000000      1.000000
25%        2.000000      0.000000      0.000000      1.000000     13.94000     16.665000     47.000000      7.001500      4.000000     36.000000     42.000000
50%        3.000000      0.000000      1.000000      1.000000     20.50000     24.240000     62.000000     12.998000     17.000000    118.000000    145.000000
75%        4.000000      0.000000      1.000000      2.000000     26.24000     31.060000     77.000000     16.997900     49.000000    222.000000    284.000000
max        4.000000      1.000000      1.000000      4.000000     41.00000     45.455000    100.000000     56.996900    367.000000    886.000000    977.000000
'''

print(train.columns)  # x--> Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp','atemp', 'humidity', 'windspeed',
#'casual', 'registered', 'count'] = y
#dtype='object')

#print(train.head(3))  #위에서부터 3개 출력
#print(train.tail())   #아래서부터 5개출력
#print(submit.columns)

x = train.drop(['datetime', 'casual','registered', 'count'], axis=1)  #컬럼(열) 삭제하려면 axis=1을 해야함.. 디폴트는 0 = 행삭제
#print(x.columns)

test_file = test_file.drop(['datetime'], axis=1)

'''
Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
       'humidity', 'windspeed'],
'''
print(x.shape)    # (10886, 8)

y = train['count']
print(y.shape)  #(10886,)

#로그변환
y = np.log1p(y)   #log는 0이 되면 안된다. 그래서 1더해줘야함! log1p는 자동으로 +1해줌

plt.plot(y)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2)모델
model = Sequential()
model.add(Dense(10, input_dim = 8))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(1))

#3)컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

#es = EarlyStopping(monitor = 'val_loss', patience = 100, mode = 'min', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4)평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2: ", r2)

rmse = RMSE(y_test, y_pred)
print("RMSE: ", rmse)

'''
loss:  1.4547288417816162
r2:  0.2576226455887306
RMSE:  1.2061214068446424
'''

results = model.predict(test_file)

submit_file['count'] = results

print(submit_file[:10])

submit_file.to_csv( path + "submitfile.csv", index=False)  # 디폴트: 기본으로 index가 생성됨 / if index= false하면 인덱스 생성x
'''
              datetime     count
0  2011-01-20 00:00:00  3.887589
1  2011-01-20 01:00:00  3.774518
2  2011-01-20 02:00:00  3.774518
3  2011-01-20 03:00:00  3.831307
4  2011-01-20 04:00:00  3.831307
5  2011-01-20 05:00:00  3.699497
6  2011-01-20 06:00:00  3.661103
7  2011-01-20 07:00:00  3.777512
8  2011-01-20 08:00:00  3.810120
9  2011-01-20 09:00:00  3.958946
'''