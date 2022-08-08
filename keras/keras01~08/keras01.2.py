from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터 
x = np.array([1,2,3,5,4])
y = np.array([1,2,3,4,5])

# 모델구성 
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=4500, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss : ', loss)
result = model.predict([6]) # 새로운 x 값을 predict한 결과 
print('6의 예측값 : ', result)

'''
loss :  0.3800458014011383
6의 예측값 :  [[5.6888285]]
'''