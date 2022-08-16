from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score

datasets = load_boston()
x = datasets.data
y = datasets.target

model = Sequential()
model.add(Dense(30, input_dim=13))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs=50, batch_size=5)

loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x)

r2 = r2_score(y, y_predict)
print('r2스코어 : ', r2)

'''
loss :  38.722633361816406
r2스코어 :  0.5413073304104685
'''