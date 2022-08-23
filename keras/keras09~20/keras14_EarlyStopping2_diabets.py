from sklearn.datasets import load_boston, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#datasets = load_boston()
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

model = Sequential()
model.add(Dense(100,input_dim=10))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor = 'val_loss', patience = 30, mode = 'min', verbose=1, restore_best_weights= True)

'''
restore_best_weights 사용
True: training이 끝난 후, model의 weight를 monitor하고 있던 값이 가장 좋았을 때의 weight로 복원함
False: 마지막 training이 끝난 후의 weight로 놔둔당..
Epoch 172/1000
315/323 [============================>.] - ETA: 0s - loss: 32.1848Restoring model weights from the end of the best epoch: 122.
323/323 [==============================] - 1s 3ms/step - loss: 31.6968 - val_loss: 39.7371
Epoch 172: early stopping
걸린시간:  195.348 초
loss:  16.705366134643555
r2스코어 :  0.8001342528779566
restore_best_weights를 사용할 시 최적의 weight값을 기록만 할 뿐 저장 기능은 없다!
저장하기 위해서는 ModelCheckpoint라는 함수는 사용해야함!
'''

start= time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[es])
end= time.time()- start

print("걸린시간: ", round(end,3), '초')

#4. 평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

plt.figure(figsize=(9,5))

#print("=========================")
#print(hist.history['loss'])
print("=========================")
print(hist.history['val_loss'])

plt.plot(hist.history['loss'], marker = '.', c='red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c='blue', label = 'val_loss')
plt.grid() #격자표시
plt.title('loss')
plt.ylabel('loss') #y축
plt.xlabel('epoch') #x축
plt.legend(loc='upper right')
plt.show()

'''
[4155.029296875, 3083.9248046875, 3319.8193359375, 3598.82421875, 3286.5908203125, 3115.7138671875, 3452.17529296875, 3157.782958984375, 3167.67724609375, 3119.285888671875, 3840.795654296875, 3370.716552734375, 3147.3017578125, 3485.421875, 3187.845947265625, 3034.87890625, 3060.656982421875, 3065.832763671875, 3332.6259765625, 3554.680419921875, 3215.110595703125, 4111.7763671875, 3161.85400390625, 3468.75341796875, 3122.724365234375, 3226.845703125, 3149.921630859375, 3224.6923828125, 3876.56298828125, 3103.005859375, 3404.320068359375, 3397.031494140625, 3037.769287109375, 3284.24609375, 3015.1904296875, 4031.962158203125, 3452.53662109375, 3153.9736328125, 3542.386474609375, 3075.16845703125, 3289.951416015625, 3129.165771484375, 3921.76416015625, 3060.885009765625, 3488.537109375, 3139.00048828125, 3103.508056640625, 3356.37841796875, 3755.36962890625, 3121.92724609375, 3147.624755859375, 3372.57177734375, 3149.402099609375, 3353.069580078125, 3328.302490234375, 3115.113037109375, 3207.209716796875, 3152.107177734375, 3089.688720703125, 3088.354248046875, 3132.968994140625, 3150.850830078125, 3675.89404296875, 3141.0576171875, 3117.619384765625]
restore_best_weights 사용
True: training이 끝난 후, model의 weight를 monitor하고 있던 값이 가장 좋았을 때의 weight로 복원함
False: 마지막 training이 끝난 후의 weight로 놔둔당..
Epoch 65/1000
270/282 [===========================>..] - ETA: 0s - loss: 2953.2510Restoring model weights from the end of the best epoch: 35.
282/282 [==============================] - 1s 3ms/step - loss: 2984.2205 - val_loss: 3117.6194
Epoch 65: early stopping
걸린시간:  67.266 초
3/3 [==============================] - 15s 7s/step - loss: 3241.3716
loss:  3241.37158203125
r2스코어 :  0.500562181492817
restore_best_weights를 사용할 시 최적의 weight값을 기록만 할 뿐 저장 기능은 없다!
'''