import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import concatenate,Concatenate
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

x1 = np.array([range(100),range(301,401)])  #2,110  삼성 저가, 고가
x2 = np.array([range(101,201),range(411,511),range(100,200)]) #3,100   미국선물 시가, 고가, 종가
x1 = np.transpose(x1) # 100,2
x2 = np.transpose(x2) # 100,3

y1 = np.array(range(1001,1101))  # 삼성전자 종가 (100,)
y2 = np.array(range(101,201))  # 하이닉스 종가  (100,)
# print(x1.shape, x2.shape, y1.shape, y2.shape)   # (100, 2) (100, 3) (100,) (100,)

x1_train, x1_test,x2_train, x2_test, y1_train, y1_test , y2_train, y2_test = train_test_split(x1,x2, y1, y2, train_size = 0.7,
                                                                                    shuffle = True, random_state = 66)
# print(x1_train.shape, x2_train.shape, y1_train.shape, y2_train.shape)  # (70, 2) (70, 3) (70,) (70,)
# print(x1_test.shape, x2_test.shape, y1_test.shape, y2_test.shape)  # (30, 2) (30, 3) (30,) (30,)

# 2-1 모델1

input1 = Input(shape= (2,))
dense1 = Dense(5, activation = 'relu', name = 'dense1')(input1)
dense2 = Dense(10, activation = 'relu', name = 'dense2')(dense1)
dense3 = Dense(20, activation = 'relu', name = 'dense3')(dense2)
output1 = Dense(7, activation = 'relu', name = 'output1')(dense3)

# 2-2 모델2

input2 = Input(shape= (3,))
dense11 = Dense(10, activation = 'relu', name = 'dense11')(input2)
dense12 = Dense(10, activation = 'relu', name = 'dense12')(dense11)
dense13 = Dense(10, name = 'dense13')(dense12)
dense14 = Dense(10, name = 'dense14')(dense13)
output2 = Dense(5, activation = 'relu', name = 'output2')(dense14)

merge1 = concatenate([output1,output2])

# 2-3 output 모델1
output21 = Dense(7)(merge1)
output22 = Dense(11)(output21)
output23 = Dense(11,activation='relu')(output22)
last_output1 = Dense(1)(output23)  # y1의 열의 갯수

# 2-4 output 모델2
output31 = Dense(7)(merge1)
output32 = Dense(21)(output31)
output33 = Dense(21)(output32)
output34 = Dense(11,activation='relu')(output33)
last_output2 = Dense(1)(output34)  # y2의 열의 갯수

model = Model(inputs=[input1, input2], outputs = [last_output1,last_output2])

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae']) # 평균절대오차

es = EarlyStopping(monitor='val_loss', patience=10, mode = 'auto', restore_best_weights=True)

model.fit([x1_train,x2_train], [y1_train,y2_train], epochs = 1000, validation_split=0.2, callbacks=[es], batch_size = 300)

#4 평가예측
loss = model.evaluate([x1_test,x2_test],[y1_test, y2_test])
print('loss: ', loss)
# loss:  [9999.42578125, 9782.2119140625, 217.21383666992188, 87.1171875, 12.323260307312012]
