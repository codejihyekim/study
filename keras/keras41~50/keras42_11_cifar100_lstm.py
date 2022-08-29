from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input, Dropout,Conv2D, Flatten,MaxPooling2D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical

# 1) 데이터
(x_train, y_train), (x_test, y_test) =cifar100.load_data()
#print(x_train.shape, y_train.shape) # 50000, 32, 32, 3) (50000, 1)
#print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

#print(np.unique(y_train)) #0~99 총 100개
n = x_train.shape[0]
x_train = x_train.reshape(n, -1)/255.

m = x_test.shape[0]
x_test = x_test.reshape(m, -1)/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

'''
# RNN은 3차원으로 
x_train = x_train.reshape(50000, 64, 48)
x_test = x_test.reshape(10000,64,48)'''

# CNN은 4차원으로
x_train= x_train.reshape(50000,32,32,3)
x_test= x_test.reshape(10000,32,32,3)

#2) 모델링
'''
# RNN
model = Sequential()
model.add(LSTM(80, input_shape = (64,48))) 
model.add(Dense(45))
model.add(Dense(100, activation='softmax'))
'''


model = Sequential()
model.add(Conv2D(25,kernel_size=(2,2),strides=1, padding='same', input_shape=(32,32,3)))
model.add(MaxPooling2D())
model.add(Conv2D(35,(3,3), activation='relu'))
model.add(Conv2D(45,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(30))
model.add(Dense(100, activation='softmax'))


#3) 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', restore_best_weights=True)
model.fit(x_train, y_train, epochs = 1000, validation_split=0.2, callbacks=[es], batch_size = 300)

#4) 평가예측
loss = model.evaluate(x_test, y_test)
print(loss)

'''
-LSTM-
loss: 3.1833 - accuracy: 0.2430
-CNN-
[2.698518991470337, 0.3402999937534332]
'''