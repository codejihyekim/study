from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.utils import to_categorical


#1) 데이터
(x_train, y_train), (x_test, y_test) =cifar100.load_data()

#print(x_train.shape, y_train.shape)    # (50000, 32, 32, 3) (50000, 1)
#print(x_test.shape, y_test.shape)      # (10000, 32, 32, 3) (10000, 1)

# print(np.unique(y_train))  # 100개

n = x_train.shape[0]
x_train = x_train.reshape(n,-1)/255.

m = x_test.shape[0]
x_test = x_test.reshape(m,-1)/255.

#x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2) 모델링
model = Sequential()
# model.add(Dense(128, input_shape = (28*28,)))
model.add(Dense(30, input_shape=(3072,)))
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(90))
model.add(Dropout(0.1))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))

#3) 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience= 70, mode = 'min', restore_best_weights=True)

model.fit(x_train, y_train, epochs = 10000, validation_split=0.2, callbacks=[es], batch_size =128)

#4) 평가예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
print("loss : ",loss[0])
print("accuracy : ",loss[1])
'''
loss :  3.324220657348633
accuracy :  0.22130000591278076
'''