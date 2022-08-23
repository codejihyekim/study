from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
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

# model.load_weights('/content/drive/MyDrive/save/keras25_1_save_weights.h5')     # loss:  67183.875
# r2스코어 :  -802.7987866662486     / 여기까지는 훈련 전의 값이 save되어있으므로 loss와 r2값이 안좋음!

# model.save("/content/drive/MyDrive/save/keras25_1_save_model.h5")
# model.save_weights("/content/drive/MyDrive/save/keras25_1_save_weights.h5")


# model.load_weights("/content/drive/MyDrive/save/keras25_3_save_weights.h5")    #compile전에 해도 이미 weights값이 들어가므로 값나오는건 상관없음!

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# start= time.time()
# hist = model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)
# end= time.time()- start

# print("걸린시간: ", round(end,3), '초')  #소수 3까지만 출력

# model.save("/content/drive/MyDrive/save/keras25_3_save_model.h5")
# model.save_weights("/content/drive/MyDrive/save/keras25_3_save_weights.h5")


# model.load_weights('/content/drive/MyDrive/save/keras25_1_save_weights.h5')    # x
model.load_weights("/content/drive/MyDrive/save/keras25_3_save_weights.h5")
# load_weights는 앞에 compile을 해줘야한다
# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
'''
loss:  21.370927810668945
r2스코어 :  0.7443146732193082
'''