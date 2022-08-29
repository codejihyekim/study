from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.metrics import r2_score
import pandas as pd

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)   # (581012, 54) (581012,)
#print(np.unique(y))    # [1 2 3 4 5 6 7]

y = pd.get_dummies(y)
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

scaler = RobustScaler() #scaler = MinMaxScaler() #scaler = StandardScaler() #scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train).reshape(len(x_train),6,3,3)
x_test = scaler.transform(x_test).reshape(len(x_test),6,3,3)

#2) 모델링

model = Sequential()
model.add(Conv2D(50,kernel_size=(2,2),strides=1,padding='same', input_shape=(6,3,3)))
model.add(Conv2D(40,kernel_size=(2,3),strides=1,padding='same'))       #5,2,10
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(30))
model.add(Dropout(0.1))
model.add(Dense(20))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))

#3) 컴파일, 훈련

model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=30, mode='min', restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=1000, validation_split=0.2, callbacks=[es])

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)
# loss :  [0.03371286392211914, 0.8445307016372681]
