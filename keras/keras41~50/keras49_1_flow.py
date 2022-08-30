from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    # vertical_flip = True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range= 5,
    zoom_range=0.1,
    # shear_range = 0.7,
    fill_mode='nearest')

augument_size = 100
# augument_size = 48 :error

# print(x_train[0].shape) # (28, 28)  2차원
# print(x_train[0].reshape(28*28).shape) # (784,)
# print(np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1).shape) # (100, 28, 28, 1)

##### flow는 x,y를 나눠줘야함 #####

x_data = train_datagen.flow(np.tile(x_train[0].reshape(28 * 28), augument_size).reshape(-1, 28, 28, 1),
                            # == x값 784 / x_train[0]을 벡터형식으로 바꾼 다음 augument_size만큼 반복하겠다.
                            np.zeros(augument_size),  # y값
                            batch_size=augument_size,
                            shuffle=False).next()

# print(type(x_data))  # <class 'tuple'>
# print(x_data)
print(x_data[0].shape, x_data[1].shape)  # (100, 28, 28, 1) (100,)

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 7))
for i in range(49):
    plt.subplot(7, 7, i + 1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')
plt.show()