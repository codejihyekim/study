from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Bidirectional,Conv1D,Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
import time
from pandas import get_dummies

# 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

#print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
#print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

#print(np.unique(y_train)) # 0~99 100개 다중분류

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(50000, 64,48)
x_test = x_test.reshape(10000,64,48)

scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1)).reshape(x_train.shape)
x_test = scaler.transform(x_test.reshape(len(x_test), -1)).reshape(x_test.shape)

# 2. 모델링
model = Sequential()
model.add(Conv1D(80,2,input_shape=(64,48)))
model.add(Flatten())
model.add(Dense(45))
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=100, mode='auto', restore_best_weights=True)
start = time.time()
model.fit(x_train, y_train, epochs=1000, validation_split=0.2, callbacks=[es], batch_size=300)
end = time.time() - start
print('걸린시간: ', round(end,3))
# 걸린시간:  56.071

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
# loss:  [3.6603598594665527, 0.17110000550746918]