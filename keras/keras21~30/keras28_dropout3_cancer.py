from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense,Input,Dropout
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import r2_score

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state=66)   #shuffle은 디폴트가 True

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2) 모델구성

input1 = Input(shape=(30,))
dense1 = Dense(120,activation='relu')(input1)
dense2 = Dense(100, activation='relu')(dense1)
dense3 = Dense(80)(dense2)
dense4 = Dense(60)(dense3)
drop1 = Dropout(0.5)(dense4)
dense5 = Dense(40)(drop1)
dense6 = Dense(20)(dense5)
drop2 = Dropout(0.2)(dense6)
output1 = Dense(1)(drop2)
model = Model(inputs=input1, outputs=output1)

#model = load_model("/content/drive/MyDrive/save/keras_03.1_cancer_save_model.h5")

model.summary()

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
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1, save_best_only=True, filepath = './_ModelCheckPoint/keras28_3_MCP.hdf5')

model.fit(x_train, y_train, epochs=10000, batch_size=1, validation_split=0.2, callbacks=[es,mcp])

model.save("/content/drive/MyDrive/save/keras_028.3_cancer_save_model.h5")

#4) 평가, 예측

print("============================1. 기본 출력=======================")

loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

print("============================2. load_model 출력=======================")

model2 = load_model('./_save/keras_028.3_cancer_save_model.h5')

loss2= model2.evaluate(x_test, y_test)
print('loss2: ', loss2)

y_predict2 = model2.predict(x_test)

r2 = r2_score(y_test, y_predict2)
print('r2스코어 : ', r2)

print("============================3. ModelCheckPoint 출력=======================")

model3 = load_model('/content/drive/MyDrive/save/keras28_3_MCP.hdf5')

loss3= model3.evaluate(x_test, y_test)
print('loss3: ', loss3)

y_predict3 = model3.predict(x_test)

r2 = r2_score(y_test, y_predict3)
print('r2스코어 : ', r2)

'''
============================1. 기본 출력=======================
6/6 [==============================] - 0s 3ms/step - loss: 0.0399
loss:  0.03992822393774986
r2스코어 :  0.8259998329121946
============================2. load_model 출력=======================
6/6 [==============================] - 0s 2ms/step - loss: 0.0399
loss2:  0.03992822393774986
r2스코어 :  0.8259998329121946
============================3. ModelCheckPoint 출력=======================
6/6 [==============================] - 0s 3ms/step - loss: 0.0264
loss3:  0.02644268609583378
r2스코어 :  0.8847674356979622
'''