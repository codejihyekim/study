import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Input, Dropout,Conv2D, Flatten
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#print(x_train.shape, y_train.shape)    # (60000, 28, 28) (60000,)  # 흑백이미지   6만장의 이미지가 28,28이다...
#print(x_test.shape, y_test.shape)      # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000,28,28,1)           #(60000,28,14,2)도 가능 / reshape= 위에 train,test의 값과 맞춰주는것!
x_test = x_test.reshape(10000,28,28,1)
#print(x_train.shape)     # (60000, 28, 28, 1)

#print(np.unique(y_train, return_counts=True))
x= x_train

y_train = to_categorical(y_train)  #one-hot 인코딩의 역할
#print(y)
#print(y_train.shape)   #(60000, 10)
y_test = to_categorical(y_test)
#print(y_test.shape)

model = Sequential()
model.add(Conv2D(50,kernel_size=(3,3), input_shape=(28,28,1)))
model.add(Conv2D(40,(3,3), activation='relu'))
model.add(Conv2D(30,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=10, mode='auto',verbose=1, restore_best_weights=False)
#mcp = ModelCheckpoint(monitor='val_loss', mode='min',verbose=1, save_best_only=True, filepath = '/content/drive/MyDrive/save/keras30_2_MCP.hdf5')

model.fit(x_train, y_train, epochs=100, batch_size=1000, validation_split=0.25, callbacks=[es])

#4)평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

'''
Epoch 64: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.2944 - accuracy: 0.9003
loss:  [0.2943989634513855, 0.9003000259399414]
'''