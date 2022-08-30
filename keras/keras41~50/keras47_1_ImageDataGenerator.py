from keras_preprocessing.image import image_data_generator
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(  # 이미지를 학습시킬 때 학습데이터의 양이 적을 경우 학습데이터를 조금씩 변형시켜서 학습데이터의 양을 늘리는 방식중에 하나
    rescale=1. / 255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest')  # 1./255(scaler역할)  , zoom_range(확대)

test_datagen = ImageDataGenerator(rescale=1. / 255)  # 평가할 때는 원래의 데이터(이미지)로 해야하므로 많은 조건을 줄 필요가 없음.(증폭할 필요 없음)

# D:\_data\image\brain

xy_train = train_datagen.flow_from_directory(
    '/tmp/rps/',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
    shuffle=True
)
# target_size = (150,150) 괄호 안의 숫자는 내 마음대로 정할 수 있다. = 맞추고 싶은 사이즈로
# "categorical"은 2D형태의 원-핫 인코딩된 라벨, "binary"는 1D 형태의 이진 라벨입니다, "sparse"는 1D 형태의 정수 라벨, "input"은 인풋 이미지와 동일한 이미지
#######Found 160 images belonging to 2 classes.#######

xy_test = test_datagen.flow_from_directory(
    '/tmp/rps-test-set/',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary'
)
#####Found 120 images belonging to 2 classes.#####

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x7f0a2a109990>
print(xy_train[0][0].shape,
      xy_train[0][1].shape)  # x = (5, 150, 150, 3) # 배치, 가로, 세로, 채널(컬러) &  y = (5,)  ==> 이진분류이므로 (5,2)로 바꿀 수 있음
print(type(xy_train))  # <class 'keras.preprocessing.image.DirectoryIterator'> 