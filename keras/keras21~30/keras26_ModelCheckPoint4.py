from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
from sklearn.metrics import r2_score

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

model = Sequential()

model.add(Dense(40, input_dim=13))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

#####################################################################################

date = datetime.datetime.now()  # 현재시간
datetime = date.strftime("%m%d_%H%M")  # string형태로 만들어랏!  %m%d_%H%M = 월일/시/분    #1206_0456
# print(datetime)

filepath = '/content/drive/MyDrive/save/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'  # 04d: 4자리까지(ex:9999),,, 4f: 소수 4제자리까지빼라     # hish에서 반환한 값이 epoch랑 val_loss로 들어가는것
model_path = "".join([filepath, 'k26_', datetime, '_', filename])  # " 구분자가 들어갑니다. ".join  <---

#   ./_ModelCheckPoint/k26_1206_0456_2500-0.3724.hdf5

########################################################################################

es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=False)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=False, filepath=model_path)

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[es, mcp])
end = time.time() - start

print("걸린시간: ", round(end, 3), '초')

model.save("/content/drive/MyDrive/save/keras26_4_save_model.h5")

# model = load_model('./_ModelCheckPoint/keras26_1_MCP.hdf5')     #ES과 ModelCheckPoint를 씀으로써 여기에는 가장 좋은 weight들이 저장됨!

# 4. 평가, 예측

print("============================1. 기본 출력=======================")

loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

print("============================2. load_model 출력=======================")

model2 = load_model('/content/drive/MyDrive/save/keras26_4_save_model.h5')

loss2 = model2.evaluate(x_test, y_test)
print('loss2: ', loss2)

y_predict2 = model2.predict(x_test)

r2 = r2_score(y_test, y_predict2)
print('r2스코어 : ', r2)

'''
============================1. 기본 출력=======================
4/4 [==============================] - 0s 3ms/step - loss: 76.3670
loss:  76.36695098876953
r2스코어 :  0.08633327095726329
============================2. load_model 출력=======================
4/4 [==============================] - 0s 3ms/step - loss: 76.3670
loss2:  76.36695098876953
r2스코어 :  0.08633327095726329
'''