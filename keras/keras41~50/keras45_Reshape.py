from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Reshape, Conv1D, LSTM
from tensorflow.keras.datasets import mnist

model = Sequential()
model.add(Conv2D(10, kernel_size=(2, 2), strides=1, padding='same', input_shape=(28, 28, 1)))
model.add(MaxPooling2D())  # (14,14,10)
model.add(Conv2D(5, (2, 2), activation='relu'))  # 13,13,5 4차원
model.add(Conv2D(7, (2, 2), activation='relu'))  # 12,12,7
model.add(Conv2D(7, (2, 2), activation='relu'))  # 11,11,7
model.add(Conv2D(10, (2, 2), activation='relu'))  # 10,10,10
model.add(Reshape((100, 10)))  # (None, 100, 10) 3차원 
model.add(Conv1D(5, 2))  # (None, 99, 5)
model.add(LSTM(15))  # Conv1D와 lstm 모두 3차원이므로 여기서는 Flatten을 안해줘도 된다. -> 2차원으로 output
model.add(Dense(10, activation='softmax'))
model.summary()

'''
Model: "sequential_18"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_5 (Conv2D)           (None, 28, 28, 10)        50        

 max_pooling2d_2 (MaxPooling  (None, 14, 14, 10)       0         
 2D)                                                             

 conv2d_6 (Conv2D)           (None, 13, 13, 5)         205       

 conv2d_7 (Conv2D)           (None, 12, 12, 7)         147       

 conv2d_8 (Conv2D)           (None, 11, 11, 7)         203       

 conv2d_9 (Conv2D)           (None, 10, 10, 10)        290       

 reshape (Reshape)           (None, 100, 10)           0         

 conv1d_10 (Conv1D)          (None, 99, 5)             105       

 lstm_3 (LSTM)               (None, 15)                1260      

 dense_33 (Dense)            (None, 10)                160       

=================================================================
Total params: 2,420
Trainable params: 2,420
Non-trainable params: 0
_________________________________________________________________
'''