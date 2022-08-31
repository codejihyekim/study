from tensorflow.python import metrics
from tensorflow.keras.datasets import imdb
import numpy as np
from tensorflow.python.keras.backend import dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten, Dropout

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000)

'''
print(x_train.shape, x_test.shape)  # (25000,) (25000,)
print(y_train.shape, y_test.shape)  # (25000,) (25000,)
print(np.unique(y_train))  # [0 1]
print(x_train[0], y_train[0])
print(type(x_train),type(y_train)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>'''

x_train = pad_sequences(x_train, padding='pre', maxlen=200, truncating='pre')
#print(x_train.shape) # (25000, 200)

x_test = pad_sequences(x_test, padding='pre', maxlen=200, truncating='pre')
#print(x_test.shape) # (25000, 200)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape) # (25000, 200) (25000, 2)
print(x_test.shape, y_test.shape) # (25000, 200) (25000, 2)

# 모델구성
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=10, input_length=200))
model.add(LSTM(32))
model.add(Dense(20))
model.add(Dropout(0.1))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(2, activation='softmax'))

# 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=300)

# 평가 예측
acc = model.evaluate(x_test, y_test)[1]
print('acc: ', acc)
# acc:  0.8367999792098999