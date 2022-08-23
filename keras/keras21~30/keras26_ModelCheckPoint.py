from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

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

# model.load_weights('/content/drive/MyDrive/save/keras25_1_save_weights.h5')

# model.save("/content/drive/MyDrive/save/keras25_1_save_model.h5")
# model.save_weights("/content/drive/MyDrive/save/keras25_1_save_weights.h5")

# model.load_weights("/content/drive/MyDrive/save/keras25_3_save_weights.h5")

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)  # restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                      filepath='/content/drive/MyDrive/save/keras26_1_MCP.hdf5')  # 다 저장하면 되므로 patience 필요 없음
# save_best_only:
# checkpoint는 Earlystopping과 쓰는게 good!
# patience 값을 많이 주면 그만큼 checkpoint를 많이 하게 된다!!
# 하지만 너무 patience값을 많이 주면 그만큼 자원낭비의 위험이 있따!
# filepath = './_ModelCheckPoint'  checkpoint를 여기에 저장하랏!
# checkpoint = 최소 loss값 저장!

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[es, mcp])
end = time.time() - start

print("=========================")
print(hist)  # 자료형
print("=========================")
print(hist.history)  # 딕셔너리 / epoch당 loss, epochs당 val
print("=========================")
print(hist.history['loss'])  # (보기편하게 : loss값)
print("=========================")
print(hist.history['val_loss'])  # (보기편하게 : val_loss값)
'''
{'loss': [2133.37255859375, 132.4376678466797, 94.16793823242188, 91.05508422851562, 93.76116180419922, 85.28755950927734,
[2133.37255859375, 132.4376678466797, 94.16793823242188, 91.05508422851562, 93.76116180419922, 85.28755950927734, 82.0737533569336,
[225.5550079345703, 91.08621978759766, 112.74756622314453, 84.01753997802734, 78.6849594116211, 82.17088317871094, 74.3660659790039,
'''

plt.figure(figsize=(9, 5))

plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()  # 격자표시
plt.title('loss')
plt.ylabel('loss')  # y축
plt.xlabel('epoch')  # x축
plt.legend(loc='upper right')
plt.show()

print("걸린시간: ", round(end, 3), '초')

model.save("/content/drive/MyDrive/save/keras26_1_save_model.h5")
# model.save_weights("/content/drive/MyDrive/save/keras25_3_save_weights.h5")

# model.load_weights('/content/drive/MyDrive/save/keras25_1_save_weights.h5')
# model.load_weights("/content/drive/MyDrive/save/keras25_3_save_weights.h5")

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
'''
걸린시간:  10.709 초
4/4 [==============================] - 0s 3ms/step - loss: 50.2750
loss:  50.27496337890625
r2스코어 :  0.39850204544613144
'''