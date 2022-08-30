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
dense2 = Dense(10, activation='relu', name='dense2')(dense1)
dense3 = Dense(20, activation='relu', name='dense3')(dense2)
output1 = Dense(7, activation='relu', name='output1')(dense3)

# 2-2 모델
input2 = Input(shape=(3,))
dense11 = Dense(50, activation='relu', name='dense11')(input2)
dense12 = Dense(10, activation='relu', name='dense12')(dense11)
dense13 = Dense(10, activation='relu', name='dense13')(dense12)
dense14 = Dense(10, activation='relu', name='dense14')(dense13)
output2 = Dense(5, activation='relu', name='output2')(dense14)

merge1 = concatenate([output1, output2])
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
loss:  0.0013765988405793905
metrics['mae']:  0.02572835236787796
r2:  0.9999984256326466
'''

'''
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_2 (InputLayer)           [(None, 3)]          0           []                               

 input_1 (InputLayer)           [(None, 2)]          0           []                               

 dense11 (Dense)                (None, 50)           200         ['input_2[0][0]']                

 dense1 (Dense)                 (None, 5)            15          ['input_1[0][0]']                

 dense12 (Dense)                (None, 10)           510         ['dense11[0][0]']                

 dense2 (Dense)                 (None, 10)           60          ['dense1[0][0]']                 

 dense13 (Dense)                (None, 10)           110         ['dense12[0][0]']                

 dense3 (Dense)                 (None, 20)           220         ['dense2[0][0]']                 

 dense14 (Dense)                (None, 10)           110         ['dense13[0][0]']                

 output1 (Dense)                (None, 7)            147         ['dense3[0][0]']                 

 output2 (Dense)                (None, 5)            55          ['dense14[0][0]']                

 concatenate (Concatenate)      (None, 12)           0           ['output1[0][0]',                
                                                                  'output2[0][0]']                

 dense_34 (Dense)               (None, 10)           130         ['concatenate[0][0]']            

 dense_35 (Dense)               (None, 7)            77          ['dense_34[0][0]']               

 dense_36 (Dense)               (None, 1)            8           ['dense_35[0][0]']               

==================================================================================================
Total params: 1,642
Trainable params: 1,642
Non-trainable params: 0
__________________________________________________________________________________________________
'''