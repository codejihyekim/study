from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

model = Sequential()
model.add(Dense(40,input_dim=13))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start= time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)
end= time.time()- start

print("걸린시간: ", round(end,3), '초')  #소수 3까지만 출력
#걸린시간:  82.281 초

model.save("/content/drive/MyDrive/save/keras25_3_save_model.h5")

#4. 평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

'''
loss:  19.45417022705078
r2스코어 :  0.7672471041615037
'''