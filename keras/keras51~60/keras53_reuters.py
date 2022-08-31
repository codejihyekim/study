from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten
import operator

(x_train,y_train), (x_test, y_test) = reuters.load_data(num_words=10000, test_split=0.2)   # num_words : 단어 사전의 갯수

#print(len(x_train), len(x_test)) #  8982   2246
#print(y_train[0])  # 3
#print(np.unique(y_train)) # 0~45까지 46개의 뉴스 카테고리

#print(type(x_train), type(y_train)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
#print(x_train.shape, y_train.shape) # (8982,) (8982,)

#print("뉴스기사의 최대길이: ", max(len(i) for i in x_train))
#print("뉴스기사의 평균길이: ", sum(map(len, x_train))/len(x_train)) #map: x_train의 len(조건)을 출력해주는 것 즉 8982의 전체 길이를 반환해준다.
'''
뉴스기사의 최대길이:  2376
뉴스기사의 평균길이:  145.5398574927633
'''

x_train = pad_sequences(x_train, padding= 'pre', maxlen= 100, truncating= 'pre') #최대길이는 2376이지만 너무 낭비이므로 평균에서 조금 줄인 100으로 하겠다.
# truncating: if maxlen보다 길이가 길면 앞에서부터 절단하겠다.
#print(x_train.shape) #(8982, 2376) --> (8982, 100)

x_test = pad_sequences(x_test, padding= 'pre', maxlen= 100, truncating= 'pre')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#print(x_train.shape, y_train.shape) # (8982, 100) (8982, 46)
#print(x_test.shape, y_test.shape) # (2246, 100) (2246, 46)

word_to_index = reuters.get_word_index()
#print(word_to_index)

index_to_word = {}
for key, value in word_to_index.items():
  index_to_word[value+3] = key
for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
  index_to_word[index] = token

print(' '.join([index_to_word[index] for index in x_train[0]]))

# 모델 구성
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=10, input_length=100))
model.add(LSTM(32))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(46, activation='softmax'))

# 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=300)

# 평가, 예측
acc = model.evaluate(x_test, y_test)[1]
print('acc: ', acc)

'''
71/71 [==============================] - 1s 4ms/step - loss: 7.5371 - acc: 0.5770
acc:  0.577025830745697
'''