from keras_preprocessing.image import image_data_generator
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip = True,
    vertical_flip = True,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    rotation_range= 5,
    zoom_range = 1.2,
    shear_range = 0.7,
    fill_mode= 'nearest')   # 1./255(scaler역할)  , zoom_range(확대)

test_datagen = ImageDataGenerator(rescale=1./255) # 평가할 때는 원래의 데이터(이미지)로 해야하므로 많은 조건을 줄 필요가 없음.(증폭할 필요 없음)

xy_train = train_datagen.flow_from_directory(
    '/tmp/rps/',
    target_size = (150,150),
    batch_size=5,
    class_mode = 'categorical',seed=66, color_mode='grayscale',
    save_to_dir='/tmp/rps/save/', shuffle= True)

xy_test = test_datagen.flow_from_directory(
    '/tmp/rps/',
    target_size = (150,150),
    batch_size = 5,
    class_mode = 'categorical')

#print(xy_train) # x와 y가 함께 있음
# <keras.preprocessing.image.DirectoryIterator object at 0x7f09a80bf7d0>

#print(xy_train[0][0])
#print(xy_train[0][1])
#print(xy_train[0][2]) # IndexError: tuple index out of range
print(xy_train[0][0].shape, xy_train[0][1].shape) # (5, 150, 150, 1) (5, 5)