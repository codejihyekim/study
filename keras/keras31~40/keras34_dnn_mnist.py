from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.utils import to_categorical
import time

#1 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_train.shape, y_train.shape)    # (60000, 28, 28) (60000,)
#print(x_test.shape, y_test.shape)      # (10000, 28, 28) (10000,)

#print(np.unique(y_train, return_counts=True))

y_train = to_categorical(y_train)
#print(y_train.shape)   #(60000, 10)
y_test = to_categorical(y_test)
#print(y_test.shape)

n = x_train.shape[0]
x_train = x_train.reshape(n,-1)/255.       # -1: 일렬로 쭉 펴주겠다.    # /255.0 이 scaler의 역할을 해주는 것과 같음!

m = x_test.shape[0]
x_test = x_test.reshape(m,-1)/255.

#2. 모델구성
model = Sequential()
#model.add(Dense(64, input_shape=(28*28, )))
model.add(Dense(80, activation = 'relu', input_shape=(784,)))
model.add(Dense(60))
model.add(Dropout(0.1))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=10, mode='auto',verbose=1, restore_best_weights=False)


start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=1000, validation_split=0.25, callbacks=[es])
end = time.time()

#4)평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)
print("걸린 시간: ", end - start)

'''
Epoch 24: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.1175 - accuracy: 0.9719
loss:  [0.11753468960523605, 0.9718999862670898]
걸린 시간:  7.85011625289917
'''