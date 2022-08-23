'''
과적합을 방지하기 위해서 학습을 진행하다가 검증 세트에서의 손실이 더 이상 감소하지 않으면
학습을 중단하는 방법을 사용할 수 있다.
케라스에서는 이를 위해 EarlyStopping이라는 콜밸함수를 제공한다.
참고로 콜백 함수는 특정 조건에서 자동으로 실행되는 함수 정도로 이해하면 된다.

ModelCheckpoint 콜백 함수
monitor 파라미터로 val_loss를 지정해서 검증 세트 손실을 기준으로 모델의 개선 여부를 판단하도록 하고,
save_best_only 파라미터를 True로 지정하여 모델이 이전에 비해 개선되었을 때만 모델을 자동으로 저장하도록 한다.
save_best_only 파라미터를 False로 설정하면 모든 에포크마다 모델을 저장한다.
verbose 파라미터를 True로 설정하여 ModelCheckpoint 콜백 함수의 동작을 모니터링 하도록 설정
이 값을 설정하면 에포크마다 모델이 개선되었는지와 자동으로 저장되었는지 여부를 알 수 있다.
'''

from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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

es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=False,
                      filepath='/content/drive/MyDrive/save/keras26_3_MCP.hdf5')

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[es, mcp])
end = time.time() - start

print("걸린시간: ", round(end, 3), '초')

model.save("/content/drive/MyDrive/save/keras26_3_save_model.h5")

# model = load_model('/content/drive/MyDrive/save/keras26_1_MCP.hdf5')     #ES과 ModelCheckPoint를 씀으로써 여기에는 가장 좋은 weight들이 저장됨!

# 4. 평가, 예측
print("============================1. 기본 출력=======================")

loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

print("============================2. load_model 출력=======================")

model2 = load_model('/content/drive/MyDrive/save/keras26_3_save_model.h5')

loss2 = model2.evaluate(x_test, y_test)
print('loss2: ', loss2)

y_predict2 = model2.predict(x_test)

r2 = r2_score(y_test, y_predict2)
print('r2스코어 : ', r2)

print("============================3. ModelCheckPoint 출력=======================")

model3 = load_model('/content/drive/MyDrive/save/keras26_3_MCP.hdf5')

loss3 = model3.evaluate(x_test, y_test)
print('loss3: ', loss3)

y_predict3 = model3.predict(x_test)

r2 = r2_score(y_test, y_predict3)
print('r2스코어 : ', r2)

'''
============================1. 기본 출력=======================
4/4 [==============================] - 0s 3ms/step - loss: 33.7353
loss:  33.73530197143555
r2스코어 :  0.5963853199772231
============================2. load_model 출력=======================
4/4 [==============================] - 0s 3ms/step - loss: 33.7353
loss2:  33.73530197143555
r2스코어 :  0.5963853199772231
============================3. ModelCheckPoint 출력=======================
4/4 [==============================] - 0s 3ms/step - loss: 33.7353
loss3:  33.73530197143555
r2스코어 :  0.5963853199772231
'''