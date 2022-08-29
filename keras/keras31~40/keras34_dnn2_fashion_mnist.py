from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# print(x_train.shape, y_train.shape)   #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)     #(10000, 28, 28) (10000,)

n = x_train.shape[0]
print(n) # 60000
x_train= x_train.reshape(n,-1)/255.

m = x_test.shape[0]
x_test = x_test.reshape(m,-1)/255.

#print(np.unique(y_train))    # [0 1 2 3 4 5 6 7 8 9]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델구성
model = Sequential()
#model.add(Dense(64, input_shape=(28*28, )))
model.add(Dense(20, input_shape=(784,)))
model.add(Dense(40))
model.add(Dense(60))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

#3) 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience= 50, mode = 'auto', restore_best_weights=True)

model.fit(x_train, y_train, epochs = 10000, validation_split=0.2, callbacks=[es], batch_size =128)

#4) 평가예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
print("loss : ",loss[0])
print("accuracy : ",loss[1])
'''
loss :  0.4201163947582245
accuracy :  0.8540999889373779
'''