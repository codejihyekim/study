import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1) 데이터
datasets = load_iris() # 유방암 데이터
#print(datasets.DESCR) # x = (150,4) y = (150,1)   ----> y를 (150,3)으로 바꿔줘야함
#print(datasets.feature_names)

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)  # (150, 4) (150,)
#print(y)
#print(np.unique(y))   # [0 1 2] = 다중분류      #이걸 해봐야 이진분류인지 다중분류인지 알 수 있움!!

y = to_categorical(y)   #One Hot Encoding
print(y)
print(y.shape)   # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


#print(x_train.shape, y_train.shape)  #(120, 4) (120, 3)
#print(x_test.shape, y_test.shape)    #(30, 4) (30, 3)

#2) 모델구성
model = Sequential()
model.add(Dense(10, activation = 'linear', input_dim = 4))
model.add(Dense(30, activation = 'linear'))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(3, activation = 'softmax'))

#3) 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])   # accuracy는 훈련에 영향 안끼친당

es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es])

#4) 평가, 예측

loss=model.evaluate(x_test, y_test)
print('loss: ', loss[0]) # 원래 [loss, accuracy] 이렇게 나오므로...
print('accuracy: ', loss[1])

results=model.predict(x_test[:7])
print(y_test[:7])
print(results)

'''
Epoch 46/100
85/96 [=========================>....] - ETA: 0s - loss: 0.1342 - accuracy: 0.9412Restoring model weights from the end of the best epoch: 36.
96/96 [==============================] - 0s 4ms/step - loss: 0.1318 - accuracy: 0.9375 - val_loss: 0.0176 - val_accuracy: 1.0000
Epoch 46: early stopping
1/1 [==============================] - 0s 173ms/step - loss: 0.0751 - accuracy: 0.9667

loss:  0.07506521046161652
accuracy:  0.9666666388511658
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[2.9872515e-04 9.9731535e-01 2.3858629e-03]
 [1.7194679e-05 9.8466969e-01 1.5313195e-02]
 [3.4126570e-05 9.6743459e-01 3.2531258e-02]
 [9.9974269e-01 2.5732885e-04 2.0162228e-18]
 [7.8242752e-05 9.9689567e-01 3.0261276e-03]
 [1.0154493e-03 9.9565351e-01 3.3310321e-03]
 [9.9981314e-01 1.8685844e-04 4.1783593e-18]]
'''