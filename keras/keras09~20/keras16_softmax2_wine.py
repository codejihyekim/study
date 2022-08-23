import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1) 데이터
datasets = load_wine() # 와인 등급
#print(datasets.DESCR) # (178, 13) (178,)  <----y=1이니까..   ----> (178, 3)으로 바꿔야 함   [0,1,2]니까 output이 3으로 바껴야 한다..
#print(datasets.feature_names)

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)  # (178, 13) (178,)
#print(y)
#print(np.unique(y))   # [0 1 2]  y를 써주는 이유: y가 출력값이므로 이진인지 다중인지 알 수 있음

y = to_categorical(y)   #One Hot Encoding
print(y)
print(y.shape)   # (178, 3)   원핫인코딩으로 3열로 바꿔줌

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#print(x_train.shape, y_train.shape)  # (142, 13) (142, 3)
#print(x_test.shape, y_test.shape)    # (36, 13) (36, 3)

#2) 모델구성
model = Sequential()
model.add(Dense(10, activation = 'linear', input_dim = 13))
model.add(Dense(30, activation = 'linear'))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(3, activation = 'softmax'))

#3) 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])   # accuracy는 훈련에 영향 안끼친당
# categorical_crossentropy은 클래스가 여러 개인 다중 분류 문제에서 사용
# 모델의 마지막 레이어의 활성화 함수는 소프트맥스 함수
es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es])

#4) 평가, 예측
loss=model.evaluate(x_test, y_test)
print('loss: ', loss[0]) # 원래 [loss, accuracy] 이렇게 나오므로
print('accuracy: ', loss[1])

results=model.predict(x_test[:7])
print(y_test[:7])
print(results)

'''Epoch 38/100
113/113 [==============================] - ETA: 0s - loss: 0.4379 - accuracy: 0.8673Restoring model weights from the end of the best epoch: 28.
113/113 [==============================] - 0s 4ms/step - loss: 0.4379 - accuracy: 0.8673 - val_loss: 0.4833 - val_accuracy: 0.8621
Epoch 38: early stopping
2/2 [==============================] - 0s 9ms/step - loss: 0.3090 - accuracy: 0.8333
loss:  0.3089979588985443
accuracy:  0.8333333134651184     ex) 개, 고양이, 사람
[[0. 0. 1.]       사람
 [0. 1. 0.]       고양이     
 [0. 1. 0.]
 [1. 0. 0.]       개
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
[[2.6074232e-04 2.8961176e-02 9.7077811e-01]
 [2.4857366e-01 5.8856714e-01 1.6285923e-01]
 [1.4708705e-04 9.9312663e-01 6.7263511e-03]
 [9.9109733e-01 6.6000377e-03 2.3026851e-03]
 [7.5725303e-04 9.5444036e-01 4.4802327e-02]
 [1.3934335e-04 9.8339581e-01 1.6464798e-02]
 [4.5735192e-02 1.7561384e-02 9.3670338e-01]]
'''