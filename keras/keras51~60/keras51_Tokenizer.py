from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text = '나는 진짜 매우 맛있는 밥을 진짜 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])    ########fit_on_text, text_to_sequences는 list형식으로 되어있다.########

print(token.word_index)
# {'진짜': 1, '마구': 2, '나는': 3, '매우': 4, '맛있는': 5, '밥을': 6, '먹었다': 7}
# Tokenizer의 word_index를 하면 많이 반복된 것부터 1번으로 배정된다.
# 어절 : 띄어쓰기 단위

x = token.texts_to_sequences([text])
print(x)  # [[3, 1, 4, 5, 6, 1, 2, 2, 7]]

word_size = len(token.word_index)
print("word_size: ", word_size)  # 7  # 총 7가지 종류가 있다.

x = to_categorical(x) # 원-핫 인코딩
print(x)
print(x.shape) # (1, 9, 8)  # 9개의 어절을 아래처럼 만들어줌(8) / to_categorical은 0부터 들어가므로 앞에 0이 있어서 8이 된다.
'''
[[[0. 0. 0. 1. 0. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 1. 0. 0. 0.]
  [0. 0. 0. 0. 0. 1. 0. 0.]
  [0. 0. 0. 0. 0. 0. 1. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 1.]]]
'''