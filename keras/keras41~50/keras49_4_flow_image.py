from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()

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
randidx = np.random.randint(x_train.shape[0], size = augment_size)
#print(x_train.shape[0]) # 60000
#print(randidx) # [53515 43863 36225 ... 21713 45505 37322]

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 1)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

x_augmented = train_datagen.flow(x_augmented, y_augmented,   #np.zeros(augument_size),
                                  batch_size= augment_size, shuffle= False).next()[0]

x_train = np.concatenate((x_train, x_augmented))  # concatenate는 괄호는 2개 써줘야한다.
y_train = np.concatenate((y_train, y_augmented))

plt.figure(figsize=(7,7))
for i in range(10):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_augmented[i], cmap='gray')
    plt.imshow(x_train[i], cmap='gray')
plt.show()