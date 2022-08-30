'''
코랩 데이터 다운로드
!wget --no-check-certificate \
https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
-O /tmp/cats_and_dogs_filtered.zip'''

'''
zip파일 열기 
import os
import zipfile

local_zip = '/tmp/cats_and_dogs_filtered.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/tmp')
zip_ref.close()
'''

from keras_preprocessing.image import image_data_generator
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,Dropout
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping


train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip = True, # 상하반전
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest') #1./255(scaler역할), zoom_range(확대)

test_datagen = ImageDataGenerator(rescale=1./255) # 평가할 때는 원래의 데이터로 해야하므로

xy_train = train_datagen.flow_from_directory(
    '/tmp/cats_and_dogs_filtered/train/',
    target_size=(100,100),
    batch_size=23,
    class_mode='categorical',
    shuffle=True
)
# target_size=(150,150) 괄호 안의 숫자는 내 마음대로 정할 수 있다.
# categorical은 2D 형태의 원-핫 인코딩된 라벨, binary는 1D 형태의 이진 라벨이다.

xy_test = test_datagen.flow_from_directory(
    '/tmp/cats_and_dogs_filtered/validation/',
    target_size=(100,100),
    batch_size=23,
    class_mode='categorical')
print(xy_train)

#2. 모델
model = Sequential()
model.add(Conv2D(128, (2,2), input_shape=(100,100,3)))
model.add(Conv2D(64,(2,2), padding='same'))
model.add(Conv2D(16,(2,2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', restore_best_weights=True)
hist = model.fit_generator(xy_train, epochs=200, steps_per_epoch=436, validation_data=xy_test, validation_steps=4, callbacks=[es])
model.save('/tmp/cats_and_dogs_filtered/save/keras48_1_save_weights1111.h5')

acc = hist.history['acc']
val_acc = hist.history['val_loss']
loss = hist.history['loss']
val_loss = hist.history['val_acc']

print('loss: ', loss[-1])
print('val_loss: ', val_loss[-1])
print('acc: ', acc[-1])
print('val_acc: ', val_acc[-1])

'''
loss:  7.15740442276001
val_loss:  0.5869565010070801
acc:  0.5299999713897705
val_acc:  0.83523029088974
'''

import pandas as pd

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

pic_path = '/tmp/cats_and_dogs_filtered/train/cats/cat.1.jpg'
model_path = '/tmp/cats_and_dogs_filtered/save/keras48_1_save_weights1111.h5'


def load_my_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(100, 100))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    if show:
        plt.imshow(img_tensor[0])
        plt.append('off')
        plt.show()

    return img_tensor


if __name__ == '__main__':
    model = load_model(model_path)
    new_img = load_my_image(pic_path)
    pred = model.predict(new_img)
    cat = pred[0][0] * 100
    dog = pred[0][1] * 100
    if cat > dog:
        print(f"당신은 {round(cat, 2)} % 확률로 고양이 입니다")
    else:
        print(f"당신은 {round(dog, 2)} % 확률로 개 입니다")
    # 당신은 71.98 % 확률로 고양이 입니다