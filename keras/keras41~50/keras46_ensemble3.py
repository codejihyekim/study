import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

x1 = np.array([range(100),range(301,401)])  #2,110  삼성 저가, 고가
#x2 = np.array([range(101,201),range(411,511),range(100,200)]) #3,100   미국선물 시가, 고가, 종가
x1 = np.transpose(x1) # 100,2
# x2 = np.transpose(x2) # 100,3

y1 = np.array(range(1001,1101))  # 삼성전자 종가 (100,)
y2 = np.array(range(101,201))  # 하이닉스 종가  (100,)
y3 = np.array(range(401,501))

#print(x1.shape, y1.shape, y2.shape, y3.shape)   # (100, 2) (100,) (100,) (100,)

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(x1,y1,y2,y3, train_size = 0.7,
                                                                                    shuffle = True, random_state = 66)
# print(x1_train.shape, y1_train.shape, y2_train.shape, y3_train.shape)  # (70, 2) (70,) (70,) (70,)
# print(x1_test.shape, y1_test.shape, y2_test.shape, y3_test.shape)  # (30, 2) (30,) (30,) (30,)

# # 2-1 모델1
input1 = Input(shape= (2,))
dense1 = Dense(5, activation = 'relu', name = 'dense1')(input1)
dense2 = Dense(10, activation = 'relu', name = 'dense2')(dense1)
dense3 = Dense(20, activation = 'relu', name = 'dense3')(dense2)
output1 = Dense(7, activation = 'relu', name = 'output1')(dense3)

# # 2-2 모델2
# input2 = Input(shape= (3,))
# dense11 = Dense(10, activation = 'relu', name = 'dense11')(input2)
# dense12 = Dense(10, activation = 'relu', name = 'dense12')(dense11)
# dense13 = Dense(10, name = 'dense13')(dense12)
# dense14 = Dense(10, name = 'dense14')(dense13)
# output2 = Dense(5, activation = 'relu', name = 'output2')(dense14)
#model = Model(inputs=input1, outputs = output1)

# 2-3 output 모델1
output21 = Dense(7)(output1)
output22 = Dense(11)(output21)
output23 = Dense(11,activation='relu')(output22)
last_output1 = Dense(1)(output23)  # y1의 열의 갯수

# 2-4 output 모델2
output31 = Dense(7)(output1)
output32 = Dense(21)(output31)
output33 = Dense(21)(output32)
output34 = Dense(11,activation='relu')(output33)
last_output2 = Dense(1)(output34)  # y2의 열의 갯수

# 2-5 output 모델3
output31 = Dense(7)(output1)
output32 = Dense(21)(output31)
output33 = Dense(21)(output32)
output34 = Dense(11,activation='relu')(output33)
last_output3 = Dense(1)(output34)  # y3의 열의 갯수

model = Model(inputs=input1, outputs = [last_output1,last_output2,last_output3])

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae']) # 평균절대오차

es = EarlyStopping(monitor='val_loss', patience=10, mode = 'auto', restore_best_weights=True)

model.fit(x1_train, [y1_train,y2_train,y3_train], epochs = 1000, validation_split=0.2, callbacks=[es], batch_size = 50)

results = model.evaluate(x1_test,[y1_test, y2_test, y3_test])
print(results)
# [1.0284323792575378e-07, 3.5762788286319847e-08, 3.7194696211884093e-09, 6.33609786859779e-08, 0.00015462239389307797, 5.009969027014449e-05, 0.0002207438083132729]

y_predict = np.array(model.predict(x1_test))
# print(y_predict.shape) # (3,30,1)
y_predict = y_predict.reshape(3,30)
r2 = r2_score([y1_test, y2_test,y3_test], y_predict)
print('r2: ', r2)
# r2:  0.9999999999997552