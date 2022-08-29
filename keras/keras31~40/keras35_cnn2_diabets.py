from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import train_test_split

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape)  # (442, 10)
print(y.shape)  # (442,)

print(np.unique(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

scaler = MinMaxScaler()

n = x_train.shape[0]
x_train_transe = scaler.fit_transform(x_train) # 2차원
#print(x_train_transe.shape)   #(353, 10)
x_train = x_train_transe.reshape(n,2,5,1) # 4차원으로 변경
# print(x_train.shape)   # (353, 2, 5, 1)

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(m,2,5,1)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(10, kernel_size=(2, 2), padding='same', input_shape=(2,5,1)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=30, mode='min', restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=256, validation_split=0.2, callbacks=[es])

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("R2 : ",r2)
'''
loss :  3213.81884765625
R2 :  0.3969155492576312
'''