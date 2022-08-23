from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import r2_score

datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.9, shuffle=True, random_state=66)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2) 모델구성(함수)

input1 = Input(shape=(10,))
dense1 = Dense(100,activation = 'relu')(input1)
dense2 = Dense(80,activation = 'relu')(dense1)
dense3 = Dense(60)(dense2)
dense4 = Dense(40)(dense3)
dense5 = Dense(20)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)

#model = load_model("/content/drive/MyDrive/save/keras_02_diabets_save_model.h5")

#model.summary()

#3) 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

# #####################################################################################
# import datetime
# date = datetime.datetime.now()   #현재시간
# datetime = date.strftime("%m%d_%H%M")  #string형태로 만들어랏!  %m%d_%H%M = 월일/시/분    #1206_0456
# #print(datetime)

# filepath = './_ModelCheckPoint/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5' #04d: 4자리까지(ex:9999),,, 4f: 소수 4제자리까지빼라     # hish에서 반환한 값이 epoch랑 val_loss로 들어가는것
# model_path = "".join([filepath, 'k27_' , datetime, '_', filename])  #" 구분자가 들어갑니다. ".join  <---

# #   ./_ModelCheckPoint/k26_1206_0456_2500-0.3724.hdf5

# ########################################################################################

es = EarlyStopping(monitor='val_loss', patience=10, mode='min',verbose=1, restore_best_weights=False)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1, save_best_only=True, filepath = '/content/drive/MyDrive/save/keras27_2_MCP.hdf5')

model.fit(x_train, y_train, epochs=10000, batch_size=5, validation_split=0.11111111, callbacks=[es,mcp])
model.save("/content/drive/MyDrive/save/keras_02.1_diabets_save_model.h5")

#4) 평가, 예측

print("============================1. 기본 출력=======================")

loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

print("============================2. load_model 출력=======================")

model2 = load_model('/content/drive/MyDrive/save/keras_02.1_diabets_save_model.h5')

loss2= model2.evaluate(x_test, y_test)
print('loss2: ', loss2)

y_predict2 = model2.predict(x_test)

r2 = r2_score(y_test, y_predict2)
print('r2스코어 : ', r2)

print("============================3. ModelCheckPoint 출력=======================")

model3 = load_model('/content/drive/MyDrive/save/keras27_2_MCP.hdf5')

loss3= model3.evaluate(x_test, y_test)
print('loss3: ', loss3)

y_predict3 = model3.predict(x_test)

r2 = r2_score(y_test, y_predict3)
print('r2스코어 : ', r2)

'''
============================1. 기본 출력=======================
2/2 [==============================] - 0s 13ms/step - loss: 3749.5894
loss:  3749.58935546875
r2스코어 :  0.5288679291203041
============================2. load_model 출력=======================
2/2 [==============================] - 0s 5ms/step - loss: 3749.5894
loss2:  3749.58935546875
r2스코어 :  0.5288679291203041
============================3. ModelCheckPoint 출력=======================
2/2 [==============================] - 0s 6ms/step - loss: 3824.4534
loss3:  3824.453369140625
r2스코어 :  0.5194613352178307
'''