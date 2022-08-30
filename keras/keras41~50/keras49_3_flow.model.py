, validation_curve
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Input, Dropout,Conv2D, Flatten
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)
#print(y_train.shape, y_test.shape) # (60000,) (10000,)

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
randidx = np.random.randint(x_train.shape[0], size = augment_size) # 랜덤한 정수값 생성하겠다.  0~ 60000만개 중에 40000만개의 값을 랜덤으로 뽑아서 변형시키겠다.(중복 포함 x) = 증폭

x_augumented = x_train[randidx].copy()  #copy() 메모리 생성
y_augumented = y_train[randidx].copy()

#print(x_augumented.shape) # (40000, 28, 28)
#print(y_augumented.shape) # (40000,)

x_augmented = x_augumented.reshape(x_augumented.shape[0], x_augumented.shape[1], x_augumented.shape[2], 1)  # (40000,28,28,1)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

x_augmented = train_datagen.flow(x_augmented, y_augumented,   #np.zeros(augument_size),
                                  batch_size= augment_size, shuffle= False).next()[0]

x_train = np.concatenate((x_train, x_augmented))  # concatenate는 괄호는 2개 써줘야한다.
y_train = np.concatenate((y_train, y_augumented))
#print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (100000, 10) (10000, 10)

# 모델링
model = Sequential()
model.add(Conv2D(50, kernel_size=(3,3), input_shape=(28,28,1)))
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

# 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto',restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=300, validation_split=0.25, callbacks=[es])

# 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_pred = model.predict(x_test)
y_pred=np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_test, y_pred)
print('acc score:', accuracy)

# loss:  [0.30613788962364197, 0.9021000266075134]