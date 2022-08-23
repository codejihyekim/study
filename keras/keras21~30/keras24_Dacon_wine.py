import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import get_dummies

path = '/content/drive/MyDrive/wine/'

train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
submit_file = pd.read_csv(path + 'sample_submission.csv')

# print(type(train))
# print(train.info())
# print(train.describe())
print(train.columns)
'''
Index(['id', 'fixed acidity', 'volatile acidity', 'citric acid',
       'residual sugar', 'chlorides', 'free sulfur dioxide',
       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'type',
       'quality'],
      dtype='object')
'''

# plt.figure(figsize=(10,10))
# sns.heatmap(data= train.corr(), square=True, annot=True, cbar=True)
# plt.show()

x = train.drop(['id','quality','citric acid','sulphates'], axis=1)
print(x.columns)

test = test.drop(['id','citric acid','sulphates'], axis=1)
y = train['quality']

le = LabelEncoder()
le.fit(x['type'])
x['type'] = le.transform(x['type'])

le.fit(test['type'])
test['type'] = le.transform(test['type'])


y = get_dummies(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=66)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test = scaler.transform(test)

#2)모델

input1 = Input(shape=(10,))
dense1 = Dense(90, activation = 'relu')(input1)
dense2 = Dense(110, activation = 'relu')(dense1)
drop1 = Dropout(0.5)(dense2)
dense3 = Dense(130, activation = 'relu')(dense2)
drop2 = Dropout(0.5)(dense3)
dense4 = Dense(150)(drop2)
drop3 = Dropout(0.3)(dense4)
dense5 = Dense(110)(drop3)
drop4 = Dropout(0.5)(dense5)
dense6 = Dense(90, activation = 'relu')(drop4)
output1 = Dense(5, activation='softmax')(drop4)
model = Model(inputs=input1, outputs=output1)

#3)컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'Nadam', metrics=['accuracy'])

es = EarlyStopping(monitor = 'val_loss', patience = 100, mode = 'min', verbose=1, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=10000, batch_size=100, validation_split=0.25, callbacks=[es])

#4)평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)
# loss:  [0.9821336269378662, 0.5853960514068604]

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

result = model.predict(test)
#print(result)
result_int = np.argmax(result, axis =1).reshape(-1,1) + 4 # 결과를 열로!
submit_file['quality'] = result_int

# argmax: 원핫인코딩된 데이터를 결과데이터에 넣을때 다시 숫자로, 되돌려 주는 편리한 기능을 제공해주는 함수 / 확률을 다시 change
acc = str(round(loss[1],4)).replace(".","_")
submit_file.to_csv( path +f"save/accuracy_{acc}.csv", index=False)