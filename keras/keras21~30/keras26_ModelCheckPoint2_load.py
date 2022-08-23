from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

'''
model = Sequential()
model.add(Dense(40,input_dim=13))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
#model.summary()
'
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min',verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1, save_best_only=True, filepath = './_ModelCheckPoint/keras26_1_MCP.hdf5')  #다 저장하면 되므로 patience 필요 없음
                                                                                 # save_best_only: 가장 좋은 지점 하나를 저장해랏
                                                                                #checkpoint는 Earlystopping과 쓰는게 good!
                                                                                 # patience 값을 많이 주면 그만큼 checkpoint를 많이 하게 된다!!
                                                                                 #하지만 너무 patience값을 많이 주면 그만큼 자원낭비의 위험이 있따!
                                                            #filepath = './_ModelCheckPoint'  checkpoint를 여기에 저장하랏!
                                                                #checkpoint = 최소 loss값 저장!

start= time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[es,mcp]) 
end= time.time()- start
print("=========================")
print(hist.history['val_loss']) 
print("걸린시간: ", round(end,3), '초')  
'''
# model.save("/content/drive/MyDrive/save/keras26_1_save_model.h5")

model = load_model(
    '/content/drive/MyDrive/save/keras26_1_MCP.hdf5')  # ES과 ModelCheckPoint를 씀으로써 여기에는 가장 좋은 weight들이 저장됨!
# model = load_model('/content/drive/MyDrive/save/keras26_1_save_model.h5')

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
'''
loss:  34.4228630065918
r2스코어 :  0.5881592051065325
'''