from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from tensorflow.keras.utils import to_categorical

datasets = load_iris()

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)  # (150, 4) (150,)
#print(np.unique(y))  # [0 1 2]

y = to_categorical(y)
#print(y)
#print(y.shape)   # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler() #scaler = StandardScaler() #scaler = RobustScaler() #scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train).reshape(len(x_train),2,2,1)
x_test = scaler.transform(x_test).reshape(len(x_test),2,2,1)

model = Sequential()
model.add(Conv2D(140, kernel_size=(3,3),padding ='same', strides=1, input_shape = (2,2,1), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(120,(2,2),padding ='same', activation='relu'))
model.add(Conv2D(100,(2,2),padding ='same', activation='relu'))
model.add(Flatten())
model.add(Dense(80))
model.add(Dropout(0.2))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(3, activation = 'softmax'))

#3. 컴파일
model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=30, mode='min', restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_split=0.2, callbacks=[es])

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("R2 : ",r2)
'''
loss :  [0.010529895313084126, 0.9666666388511658]
R2 :  0.9535907353056635
'''