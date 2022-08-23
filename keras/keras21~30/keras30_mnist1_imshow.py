import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)    # (60000, 28, 28) (60000,)  # 흑백이미지   6만장의 이미지가 28,28
print(x_test.shape, y_test.shape)      # (10000, 28, 28) (10000,)

print(x_train[0])            # 0은 흰색 / 숫자는 밝기(진하기)   === 숫자 5모양
print('y_train[0]번째 값 : ',y_train[0])
# y_train[0]번째 값 :  5

plt.imshow(x_train[0], 'gray')
plt.show()