import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as keras_image

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    vertical_flip= True,
    width_shift_range = 0.1,
    height_shift_range= 0.1,
    rotation_range= 5,
    zoom_range = 1.2,
    shear_range=0.7,
    fill_mode = 'nearest',
    validation_split=0.3)

train_generator = train_datagen.flow_from_directory(
    '/tmp/rps/',
    target_size=(100,100),
    batch_size=10,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    '/tmp/rps/', # same directory as training data
    target_size=(100,100),
    batch_size=10,
    class_mode='categorical',
    subset='validation') # set as validation data

#print(train_generator[0][0].shape)  # (10, 100, 100, 3)
#print(validation_generator[0][0].shape) # (10, 100, 100, 3)

test_datagen = ImageDataGenerator(rescale = 1./255)

np.save('/tmp/rps/save/keras48_2_train_x.npy', arr = train_generator[0][0])
np.save('/tmp/rps/save/keras48_2_train_y.npy', arr = train_generator[0][1])
np.save('/tmp/rps/save/keras48_2_test_x.npy', arr = validation_generator[0][0])
np.save('/tmp/rps/save/keras48_2_test_y.npy', arr = validation_generator[0][1])

#print(train_generator[0])
#print(validation_generator[0])

## np.save('/tmp/rps/save/keras48_2_train_x.npy', arr = train_generator[0][0])
## np.save('/tmp/rps/save/keras48_2_train_y.npy', arr = train_generator[0][1])
## np.save('/tmp/rps/save/keras48_2_test_x.npy', arr = validation_generator[0][0])
## np.save('/tmp/rps/save/keras48_2_test_y.npy', arr = validation_generator[0][1])

model = Sequential()
model.add(Conv2D(32,(2,2), input_shape = (100,100,3)))
model.add(Flatten())
model.add(Dense(72,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(3,activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es=EarlyStopping(monitor='val_loss', patience=50, mode='auto', restore_best_weights=True)
hist = model.fit_generator(train_generator, epochs = 1000, steps_per_epoch = 252,
                    validation_data = validation_generator,
                    validation_steps = 4, callbacks=[es])
model.save('/tmp/rps/save/keras48_3_save_weights1111.h5')

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
print('val_acc:',val_acc [-1])

sample_directory = '/tmp/rps-test-set/paper/'
sample_image = sample_directory + 'testpaper01-00.png'

# 샘플 케이스 확인
image_ = plt.imread(str(sample_image))
plt.title("Test Case")
plt.imshow(image_)
plt.axis('off')
plt.show()

# 샘플케이스 평가
loss, acc = model.evaluate(validation_generator)
print("Between R.S.P Accuracy : ", str(np.round(acc, 2)*100) + "%")
image_ = keras_image.load_img(str(sample_image), target_size=(100,100))
x = keras_image.img_to_array(image_)
x = np.expand_dims(x, axis=0)
x /= 255.
images = np.vstack([x])
classes = model.predict(images, batch_size=40)
y_predict = np.argmax(classes)#NDIMS

validation_generator.reset()
print(validation_generator.class_indices)

if(y_predict==0):
    print(classes[0][0]*100, "의 확률로")
    print(" → '보'입니다. " )
elif(y_predict==1):
    print(classes[0][1]*100, "의 확률로")
    print(" → '바위'입니다. ")
elif(y_predict==2):
    print(classes[0][2]*100, "의 확률로")
    print(" → '가위'입니다. ")
else:
    print("ERROR")
'''
loss: 9598.396484375    
val_loss: 27864.55078125
acc: 0.15646257996559143
val_acc: 0.0
Between R.S.P Accuracy :  0.0%
{'rps': 0}
100.0 의 확률로
 → '바위'입니다.
'''