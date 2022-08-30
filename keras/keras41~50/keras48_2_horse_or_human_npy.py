from keras_preprocessing.image import image_data_generator
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping

# 1. 데이터
train_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

xy_train = train_datagen.flow_from_directory(
    '/tmp/horse-or-human/',
    target_size=(100, 100),
    batch_size=50,
    class_mode='binary',
    shuffle=True)
xy_test = test_datagen.flow_from_directory(
    '/tmp/horse-or-human/',
    target_size=(100, 100),
    batch_size=50,
    class_mode='binary')
print(xy_train[0][0].shape, xy_train[0][1].shape)  # (50, 100, 100, 3) (5,)

'''np.save('/tmp/horse-or-human/save/keras48_2_train_x.npy', arr = xy_train[0][0])   
np.save('/tmp/horse-or-human/save/keras48_2_train_y.npy', arr = xy_train[0][1])
np.save('/tmp/horse-or-human/save/keras48_2_test_x.npy', arr = xy_test[0][0])
np.save('/tmp/horse-or-human/save/keras48_2_test_y.npy', arr = xy_test[0][1])'''

x_train = np.load('/tmp/horse-or-human/save/keras48_2_train_x.npy')
y_train = np.load('/tmp/horse-or-human/save/keras48_2_train_y.npy')
x_test = np.load('/tmp/horse-or-human/save/keras48_2_test_x.npy')
y_test = np.load('/tmp/horse-or-human/save/keras48_2_test_y.npy')
print(x_train.shape)  # (50, 100, 100, 3)
print(y_train.shape)  # (50,)

# 2. 모델링
model = Sequential()
model.add(Conv2D(30, (2, 2), input_shape=(100, 100, 3)))
model.add(Conv2D(20, (2, 2)))
model.add(Conv2D(10, (2, 2)))
model.add(Flatten())
model.add(Dense(20))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=100, mode='auto', restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, validation_split=0.2, callbacks=[es], batch_size=300)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

# loss:  [-6393932808192.0, 0.5199999809265137]