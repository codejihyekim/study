from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical

#1) 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
#print(x.shape, y.shape) # (178, 13) (178,)
#print(np.unique(y)) # [0 1 2]

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, shuffle=True, random_state=66)
print(x_train.shape, x_test.shape) # (124, 13) (54, 13)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train).reshape(124,13,1)
x_test = scaler.transform(x_test).reshape(54,13,1)

#2. 모델링
model = Sequential()
model.add(LSTM(80, input_shape=(13,1)))
model.add(Dense(45))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', restore_best_weights=True)
model.fit(x_train, y_train,epochs=1000, validation_split=0.2, callbacks=[es], batch_size=128)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
# loss:  [0.40666934847831726, 0.8518518805503845]