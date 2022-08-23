from sklearn.datasets import load_boston, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#datasets = load_boston() # 회귀 분석용 보스턴 집값
datasets = load_diabetes() # 회귀 분석용 당뇨병 자료
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

model = Sequential()
model.add(Dense(100,input_dim=10))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start= time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)
end= time.time()- start

print("걸린시간: ", round(end,3), '초')

#4. 평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

'''
loss:  3314.1279296875
r2스코어 :  0.4893517690891931
'''
'''
print("=========================")
print(hist)  #자료형
print("=========================")
print(hist.history)  #딕셔너리 / epoch당 loss, epochs당 val
print("=========================")
print(hist.history['loss'])  #(보기편하게 : loss값)
print("=========================")
print(hist.history['val_loss']) #(보기편하게 : val_loss값)
'''

plt.figure(figsize=(9,5))

plt.plot(hist.history['loss'], marker = '.', c='red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c='blue', label = 'val_loss')
plt.grid() #격자표시
plt.title('loss')
plt.ylabel('loss') #y축
plt.xlabel('epoch') #x축
plt.legend(loc='upper right') #
plt.show()
'''
loss와 val_loss 간의 간격이 넓으면 과적합 
val_loss가 loss보다 낮으면 good
if 시각해서 봤을 때 중간에 val_loss가 좋은(낮음) 지점이 있다면 거기까지만 돌리면됨 
hist.history (일단 몇번의 기회를 주고 돌림 -> 그 간격 중에 최저점이 있으면 그 지점부터 다시 몇번을 돌림 -> (계속반복)-> 새로운 값이 갱신 안되면 그 찍었던 최저점에서 끝)
'''

'''
=========================
<tensorflow.python.keras.callbacks.History object at 0x0000016228CB5700>
=========================
{'loss': [2516.3447265625, 172.50013732910156, 115.53948211669922, 108.63299560546875, 93.22905731201172, 86.24969482421875, 77.85194396972656, 83.2345962524414, 72.87215423583984, 75.53510284423828], 'val_loss': [93.7500228881836, 87.81674194335938, 253.3147735595703, 77.14508819580078, 77.12211608886719, 72.41887664794922, 75.14170837402344, 77.72003173828125, 141.56031799316406, 87.57982635498047]}
=========================
[2516.3447265625, 172.50013732910156, 115.53948211669922, 108.63299560546875, 93.22905731201172, 86.24969482421875, 77.85194396972656, 83.2345962524414, 72.87215423583984, 75.53510284423828]
=========================
[93.7500228881836, 87.81674194335938, 253.3147735595703, 77.14508819580078, 77.12211608886719, 72.41887664794922, 75.14170837402344, 77.72003173828125, 141.56031799316406, 87.57982635498047]'''
