from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target  # y: 0 or 1
#print(x.shape, y.shape)   # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MaxAbsScaler()

scaler.fit(x_train)  # 어느 비율로 나눌지
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) # x_train에 맞는 비율로 들어가있움

model = Sequential()
model.add(Dense(10, input_dim = 30))
model.add(Dense(20))
model.add(Dense(30,activation = 'relu'))
model.add(Dense(40,activation = 'relu'))
model.add(Dense(50))
model.add(Dense(1, activation = 'sigmoid'))

#model.summary()

#3) 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

# es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', verbose=1, restore_best_weights=True) # auto, max, min

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2) #callbacks=[es])

#4) 평가, 예측

#회귀모델의 보조지표로 r2를 썼다면 이진분류에서는 필요가 없다..
loss=model.evaluate(x_test, y_test)
print('loss: ', loss) # index 0 은 loss, index 1은 accuracy

'''
# MinMaxScaler
loss:  [0.27388572692871094, 0.9649122953414917]
# StandardScaler
loss:  [0.15372048318386078, 0.9736841917037964]
# RobustScaler
loss:  [0.3649604618549347, 0.9561403393745422]
# MaxAbsScaler
loss:  [0.5417976975440979, 0.9385964870452881]
'''