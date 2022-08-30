import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import concatenate, Concatenate  # Concatenate = 그냥 이어주는 것(합쳐주는 것)
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

x1 = np.array([range(100), range(301, 401)])  # 2,100 삼성 저가, 고가
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])  # 3, 100 미국선물 시가, 고가, 종가
x1 = np.transpose(x1)  # 100, 2
x2 = np.transpose(x2)  # 100, 3

y = np.array(range(1001, 1101))  # 삼성전자 종가
# print(x1.shape, x2.shape, y.shape) # (100, 2) (100, 3) (100,)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.7, shuffle=True,
                                                                         random_state=66)
print(x1_train.shape, x2_train.shape, y_train.shape)  # (70, 2) (70, 3) (70,)
print(x1_test.shape, x2_test.shape, y_test.shape)  # (30, 2) (30, 3) (30,)

# 2-1 모델
input1 = Input(shape=(2,))
dense1 = Dense(5, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(7, activation='relu', name='dense3')(dense2)
output1 = Dense(7, activation='relu', name='output1')(dense3)

# 2-2 모델
input2 = Input(shape=(3,))
dense11 = Dense(30, activation='relu', name='dense11')(input2)
dense12 = Dense(20, activation='relu', name='dense12')(dense11)
dense13 = Dense(40, name='dense13')(dense12)
dense14 = Dense(50, name='dense14')(dense13)
output2 = Dense(5, activation='relu', name='output2')(dense14)

merge1 = Concatenate(axis=-1)([output1, output2])
merge2 = Dense(10, activation='relu')(merge1)
merge3 = Dense(7)(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mae')
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', restore_best_weights=True)
model.fit([x1_train, x2_train], y_train, epochs=1000, validation_split=0.2, callbacks=[es], batch_size=50)

# 4. 평가 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss: ', loss[0])
print("metrics['mae']: ", loss[1])

y_predict = model.predict([x1_test, x2_test])

r2 = r2_score(y_test, y_predict)
print('r2: ', r2)

'''
loss:  2.0265579792067e-07
metrics['mae']:  0.0004435221489984542
r2:  0.9999999997682297
'''
