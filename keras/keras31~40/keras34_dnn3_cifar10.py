from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1) 데이터

(x_train, y_train),(x_test,y_test) = cifar10.load_data()

#print(x_train.shape, y_train.shape)
#print(x_test.shape, y_test.shape)

#print(np.unique(y_train))  # [0 1 2 3 4 5 6 7 8 9]

n = x_train.shape[0]
x_train = x_train.reshape(n,-1)/255.           # /255.0 이 scaler의 역할을 해주는 것과 같음!

m = x_test.shape[0]
x_test = x_test.reshape(m,-1)/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2) 모델링
model = Sequential()
# model.add(Dense(128, input_shape = (28*28,)))
model.add(Dense(100, input_shape=(3072,)))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(60, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#3) 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience= 50, mode = 'min', restore_best_weights=True)

model.fit(x_train, y_train, epochs = 10000, validation_split=0.2, callbacks=[es], batch_size =128)

#4) 평가예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
print("loss : ",loss[0])
print("accuracy : ",loss[1])
'''
loss :  1.4242994785308838
accuracy :  0.4957999885082245
'''