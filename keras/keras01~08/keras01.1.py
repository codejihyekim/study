from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터 
# x와 y의 array를 정의, array안에 리스트로 작성한다.
x =  np.array([1,2,3])
y =  np.array([1,2,3])

# 모델구성 
model = Sequential() #Sequential은 레이어를 선형으로 연결하여 구성한다. 
model.add(Dense(1, input_dim=1)) # add로 층을 추가 첫번째 인자는 출력 뉴력의 수, input_dim은 입력 뉴런의 수 
# 2D 계층은 입력 모양을 input_dim으로 설정할 수 있다. 

#3. 컴파일, 훈련
# mse는 실제 정답에 대한 정답률의 오차뿐만 아니라 다른 오답들에 대한 정답률 오차 또한 포함하여 계산
# adam은 진행하던 속도에 관성을 주고, 최근 경로의 곡면의 변화량에 따른 적응적 학습률을 갖는 알고리즘 
model.compile(loss='mse', optimizer='adam') # 평균 제곱 에러 mse 이 값은 작을수록 좋다. optimizer='adam'은 mse 값을 감축시키는 역할 
model.fit(x, y, epochs=4200, batch_size=1)  # 모델을 학습시키기 위해서 fit() 함수를 사용
# batch_size 값이 크면 클수록 여러 데이터를 기억하고 있어야 하기 때문에 메모리가 커야 한다. 
# batch_size 값이 작으면 학습은 꼼꼼하게 이루어지지만 시간이 많이 걸린다. 

#4. 평가, 예측
loss = model.evaluate(x, y) # loss는 예측값과 실제값이 차이나느 정도를 나타내는 지표 
print('loss : ', loss)
result = model.predict([4]) # 학습이 완료된 모델에 검증 데이터를 넣으면 예측 값을 반환한다. 
print('4의 예측값 : ', result)

'''
loss :  1.3554473810017953e-07
4의 예측값 :  [[3.999242]]
'''

