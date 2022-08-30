from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Bidirectional,Conv1D,Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import time

# 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

#print(x.shape, y.shape) # (506, 13) (506,)

x = x.reshape(506,13,1)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, shuffle=True, random_state=66)
print(x_train.shape, x_test.shape) # (354, 13, 1) (152, 13, 1)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1)).reshape(x_train.shape)
x_test = scaler.transform(x_test.reshape(len(x_test), -1)).reshape(x_test.shape)

print(x_train.shape, x_test.shape) # (354, 13, 1) (152, 13, 1)

# 모델링
#-Conv1D-
model = Sequential()
model.add(Conv1D(80,2,input_shape = (13,1)))
model.add(Flatten())
model.add(Dense(45))
model.add(Dense(1))

'''
#-LSTM-
model = Sequential()
model.add(LSTM(80, input_shape = (13,1))) 
model.add(Dense(45))
model.add(Dense(1))'''

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode = 'auto', restore_best_weights=True)

start = time.time()
model.fit(x_train, y_train, epochs = 1000, validation_split=0.2, callbacks=[es], batch_size = 300)
end = time.time() - start
print('걸린시간: ', round(end, 3))
# 걸린시간:  11.871

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2: ', r2)
'''
-Conv1D-
걸린시간:  11.871
loss:  18.627885818481445
r2:  0.7745274761185165

-LSTM-
걸린시간:  27.08
loss:  16.553518295288086
r2:  0.799635674131646
'''