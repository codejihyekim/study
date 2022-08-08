from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#모델구성 layer와 parameter를 추가하여 deep 러닝으로 만들어본다. 
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3)) # 위에서 나온 출력이 그대로 다음 layer에 넘어가기 때문에 다음 layer에는 input 개수를 안써도 된다.
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
# 모델층을 두껍게해서 다중신경망을 형성하여 그 뒤에 컴파일하고 예측을 하면 
# 단일신경망일때에 비해 훈련량 epochs를 훨씬 줄여도 loss 값을 구할 수 있다. 


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss : ', loss)
result = model.predict([4])
print('4의 예측값 : ', result)

'''
loss :  0.0005467088194563985
4의 예측값 :  [[3.9553618]]
'''

