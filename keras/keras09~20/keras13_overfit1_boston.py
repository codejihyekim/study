from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

model = Sequential()
model.add(Dense(100,input_dim=13))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start= time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=1, validation_split=0.2)
end= time.time()- start

print("걸린시간: ", round(end,3), '초')  #소수 3까지만 출력

#4. 평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

'''
loss:  55.287635803222656
r2스코어 :  0.3385296197696094
'''

print("=========================")
print(hist)  #자료형
print("=========================")
print(hist.history)  #딕셔너리 / epoch당 loss, epochs당 val
print("=========================")
print(hist.history['loss'])  #(보기편하게 : loss값)
print("=========================")
print(hist.history['val_loss']) #(보기편하게 : val_loss값)

plt.figure(figsize=(9,5))

plt.plot(hist.history['loss'], marker = '.', c='red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c='blue', label = 'val_loss')
plt.grid() #격자표시
plt.title('loss')
plt.ylabel('loss') #y축
plt.xlabel('epoch') #x축
plt.legend(loc='upper right')
plt.show()