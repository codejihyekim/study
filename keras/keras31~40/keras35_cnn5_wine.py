from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense,Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.metrics import r2_score
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns    # matplot보다 이뿌게

#1) 데이터
datasets = load_wine()

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)  # (178, 13) (178,)
#print(y)
#print(np.unique(y))  # [0 1 2]

print(datasets.feature_names)  #컬럼 이름
# print(datasets.DESCR)
'''
['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
'''

xx = pd.DataFrame(x, columns= datasets.feature_names)  # x가 pandas로 바껴서 xx에 저장, columns를 칼럼명이 나오게 지정해준다.
#print(type(xx))
#print(xx)

xx['result'] = y

print(xx.corr())  # 상관관계

plt.figure(figsize=(10,10))
sns.heatmap(data = xx.corr(), square = True, annot=True, cbar=True)   #cbar = 옆에 컬러
plt.show()

x = xx.drop(['ash','result'],axis=1)
x = x.to_numpy()    #(178,12)
#print(x)

y = to_categorical(y)
#print(y)
#print(y.shape)   # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

scaler = MaxAbsScaler() #scaler = MinMaxScaler() #scaler = StandardScaler() #scaler = RobustScaler()

x_train = scaler.fit_transform(x_train).reshape(len(x_train),3,4,1)
x_test = scaler.transform(x_test).reshape(len(x_test),3,4,1)

#2) 모델링
model = Sequential()
model.add(Conv2D(10,kernel_size=(2,2),strides=1,padding='same', input_shape=(3,4,1)))
model.add(Conv2D(20,kernel_size=(2,3),strides=1,padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(30))
model.add(Dropout(0.1))
model.add(Dense(40))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

#3) 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=30, mode='min', restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_split=0.2, callbacks=[es])

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("R2 : ",r2)

'''
loss :  0.00390927167609334
R2 :  0.9836055418245335
'''