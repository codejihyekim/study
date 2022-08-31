from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten, Conv1D  # Embedding: 좌표로 change

# 1. 데이터
docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요', '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글쎄요',  # x (13,?)
        '별로에요', '생각보다 지루해요', '연기가 어색해요', ' 재미없어요', '너무 재미없다', '참 재밌네요', '선생님이 잘 생기긴 했어요']

labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1])  # (13,) [0,1] = 이진분류
# print(labels.shape)

token = Tokenizer()
token.fit_on_texts(docs)
# print(token.word_index)
# {'참': 1, '너무': 2, '잘': 3, '재밌어요': 4, '최고에요': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14,
# '싶네요': 15, '글쎄요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '선생님이': 25, '생기긴': 26, '했어요': 27}

x = token.texts_to_sequences(docs)
# print(x)
# [[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]
### shape 맞춰주려면 가장 긴 것을 기준으로 공백을 채워주면된다.
### 공백을 채울 때 보통 0을 써주는데 중간에 넣어주면 안된다.
### 앞이나 뒤에 채워주면 되는데 통상적으로 앞에 채워준다.

pad_x = pad_sequences(x, padding='pre', maxlen=5)  # 최대길이에 맞추려고
# print(pad_x)
# print(pad_x.shape) # (13, 5)

'''
[[ 0  0  0  2  4] 기존 x[0] -> [2, 4]이 바꿔짐 
 [ 0  0  0  1  5]
 [ 0  1  3  6  7]
 [ 0  0  8  9 10]
 [11 12 13 14 15]
 [ 0  0  0  0 16]
 [ 0  0  0  0 17]
 [ 0  0  0 18 19]
 [ 0  0  0 20 21]
 [ 0  0  0  0 22]
 [ 0  0  0  2 23]
 [ 0  0  0  1 24]
 [ 0 25  3 26 27]]
'''

word_size = len(token.word_index)
# print('word_size: ', word_size)  # word_size:  27
# print(np.unique(pad_x))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27]

pad_x = pad_x.reshape(13,5,1) # 3차원으로 바꿈

# 모델
model = Sequential()
model.add(LSTM(32, input_shape=(5,1)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=100)

# 평가, 예측
acc = model.evaluate(pad_x,labels)[1]
print('acc: ', acc)
# acc:  0.9230769276618958