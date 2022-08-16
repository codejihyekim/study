from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
         train_size=0.8, shuffle=True, random_state=49)

model = Sequential()
model.add(Dense(90, input_dim=10))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=50, batch_size=10)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

'''
loss :  2095.9453125
r2스코어 :  0.6066399281709276
'''