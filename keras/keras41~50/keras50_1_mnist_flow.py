from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Conv2D,Flatten
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train, y_train),(x_test, y_test) = mnist.load_data()
#print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
#print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip = True,
    #vertical_flip = True,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    #rotation_range= 5,
    zoom_range = 0.1,
    #shear_range = 0.7,
    fill_mode= 'nearest')

augment_size = 40000
#print(x_train[0].shape) # (28, 28)
#print(x_train[0].reshape(28*28).shape) # (784,)

randidx = np.random.randint(x_train.shape[0], size = augment_size)

x_augmented = x_train[randidx].copy()  #copy() 메모리 생성
y_augmented = y_train[randidx].copy()

x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 1)
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

x_augmented = train_datagen.flow(x_augmented, y_augmented,   #np.zeros(augument_size),
                                  batch_size= augment_size, shuffle= False, save_to_dir='/tmp/').next()[0]

x_train = np.concatenate((x_train, x_augmented))  # concatenate는 괄호는 2개 써줘야한다.
y_train = np.concatenate((y_train, y_augmented))

#2. 모델
model = Sequential()
model.add(Conv2D(64,(2,2), input_shape= (28,28,1)))
model.add(Conv2D(64,(2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss= 'sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=50, mode='auto',verbose=1, restore_best_weights=False)

model.fit(x_train, y_train, epochs=100, batch_size=1000, validation_split=0.25, callbacks=[es])
'''
Epoch 51: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.3624 - acc: 0.9139
'''

#4)평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)
# loss:  [0.3623749613761902, 0.9139000177383423]