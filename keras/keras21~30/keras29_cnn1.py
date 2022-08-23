from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D  # Convolution

model = Sequential()
model.add(Conv2D(10, kernel_size=(2, 2), input_shape=(5, 5, 1)))  # ===> (N,4,4,10)이 된다
model.add(Conv2D(5, (2, 2), activation='relu'))  # ===> (3,3,5)가 된다
model.add(Conv2D(7, (2, 2), activation='relu'))  # ===> (2,2,7)가 된다
# 이미지  /  kernel_size = 자르는 사이즈 / 가로 x 세로 x 장수<컬러> (장수가 1이므로 흑백사진)
# Conv2D는 무조건 3차원


# strides =1 이 디폴트
# padding = same : 둘러싼다 / padding = valid가 디폴트   : ex) 사진테두리 부분은 한번밖에 못돌리므로 padding을 써줌으로써 테두리부분도 여러번 돌릴수 있게 해준다!
# maxpooling = conv 다음에 해줘야함! conv해준거를 토대로 값이 높은 값들만 뽑아서 다음 conv로 넘겨준다
# 그래서 maxpooling을 하고 나면 반으로 훅 준다!  ((2,2)가 디폴트)   ---> 줄여주면 속도도 더 빨라진다는 장점!

# model = Sequential()
# model.add(Conv2D(10, kernel_size=(5,5), strides=1,
#                  padding='same', input_shape=(10, 10, 1)))
# model.add(MaxPooling2D(3))    # ((2,2)가 디폴트)   = dropout가 비슷
# model.add(Conv2D(5, (2,2), activation = 'relu'))     # 7,7,5
# model.add(Conv2D(7, (2,2), activation = 'relu'))       # 6,6,7
# model.add(Flatten())    # 4차원에서 2차원으로 만들어주려고 펴주는 기능   #(5,1)로 나와야하니까..
# model.add(Dense(64))
# model.add(Dropout(0.2))
# model.add(Dense(16))
# model.add(Dense(5, activation = 'softmax'))

model.summary()

'''
Model: "sequential_5"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_7 (Conv2D)           (None, 4, 4, 10)          50        

 conv2d_8 (Conv2D)           (None, 3, 3, 5)           205       

 conv2d_9 (Conv2D)           (None, 2, 2, 7)           147       

=================================================================
Total params: 402
Trainable params: 402
Non-trainable params: 0
_________________________________________________________________



Model: "sequential_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_5 (Conv2D)           (None, 10, 10, 10)        260       

 max_pooling2d_2 (MaxPooling  (None, 3, 3, 10)         0         
 2D)                                                             

 conv2d_6 (Conv2D)           (None, 2, 2, 5)           205       

=================================================================
Total params: 465
Trainable params: 465
Non-trainable params: 0
_________________________________________________________________
'''