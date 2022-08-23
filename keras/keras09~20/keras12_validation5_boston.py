from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

datasets = load_boston()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape)
print(y.shape)

print(datasets.feature_names)  #컬럼 이름
print(datasets.DESCR)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_test,y_test, train_size=0.5, random_state=66)

model = Sequential()
model.add(Dense(50, input_dim=13))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs=200, batch_size=1, validation_data=(x_val, y_val))

loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x)

r2 = r2_score(y, y_predict)
print('r2스코어 : ', r2)

'''
loss :  25.30260467529297
r2스코어 :  0.7002755536294962
'''