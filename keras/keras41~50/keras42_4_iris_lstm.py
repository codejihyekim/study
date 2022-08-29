from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical

#1) 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (150, 4) (150,)

y = to_categorical(y)
# print(y)
# print(y.shape) # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, shuffle=True, random_state=66)

print(x_train.shape, x_test.shape) # (105, 4) (45, 4)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train).reshape(105,4,1)
x_test = scaler.transform(x_test).reshape(45,4,1)

# 모델링
model = Sequential()
model.add(LSTM(80, input_shape=(4,1)))
model.add(Dense(45))
model.add(Dense(3, activation='softmax'))

# 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, validation_split=0.2, callbacks=[es], batch_size=128)

#평가 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
# loss:  [0.13714061677455902, 0.9555555582046509]