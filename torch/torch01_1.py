import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

x = torch.FloatTensor(x).unsqueeze(1)  # tensor형으로 바꿔줘야함 / unsqueeze 해줌으로써 (3,) -> (3,1)이 됨
y = torch.FloatTensor(y).unsqueeze(1)  # unsqueeze 해줌으로써 (3,) -> (3,1)이 됨

print(x, y)  # 스칼라가 3개
print(x.shape, y.shape)  # torch.Size([3]) torch.Size([3]) -> torch.Size([3, 1]) torch.Size([3, 1])
'''
tensor([[1.],
        [2.],
        [3.]]) tensor([[1.],
        [2.],
        [3.]])
torch.Size([3, 1]) torch.Size([3, 1])
'''

# 모델구성
# model = Sequential()
# model.add(Dense(1,input_dim=1))
model = nn.Linear(1, 1)  # torch에서는 input을 맨 앞에, 그 뒤에는 output을 써줌
# x의 1(앞), y의 1(뒤)

# 3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

criterion = nn.MSELoss()  # 평가지표의 표준
optimizer = optim.Adam(model.parameters(), lr=0.01)  # model.parameters()은 어떤 모델을 엮을것인지 즉 model = nn.Linear
# optimizer = optim.SGD(model.parameters(), lr = 0.01)


print(optimizer)
''' 
Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    eps: 1e-08
    foreach: None
    lr: 0.01
    maximize: False
    weight_decay: 0
)
SGD (
Parameter Group 0
    dampening: 0
    foreach: None
    lr: 0.01
    maximize: False
    momentum: 0
    nesterov: False
    weight_decay: 0
)
'''


# model.fit(x ,y , epochs=10000, batch_size=1)

def train(model, criterion, optimizer, x, y):
    # model.train()  # 훈련모드
    optimizer.zero_grad()  # 기울기 초기화

    hypothesis = model(x)  # x를 넣었을 때의 값이 hypothesis에 담김 # y = wx + b

    loss = criterion(hypothesis, y)  # 예측값과 실제값 비교 # MSE
    # 여기까지가 순전파

    loss.backward()  # 기울기값 계산까지
    optimizer.step()  # 가중치 수정(역전파)
    return loss.item()


epochs = 100
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch : {}, loss: {}'.format(epoch, loss))
'''
optimizer -> SGD
epoch : 96, loss: 4.353107669885503e-06
epoch : 97, loss: 4.332465778134065e-06
epoch : 98, loss: 4.3112972889503e-06
epoch : 99, loss: 4.290692231734283e-06
epoch : 100, loss: 4.269736109563382e-06
optimizer -> Adam
epoch : 96, loss: 0.039895761758089066
epoch : 97, loss: 0.039716873317956924
epoch : 98, loss: 0.03953789547085762
epoch : 99, loss: 0.0393591932952404
epoch : 100, loss: 0.039181020110845566
'''
print("==========================================")


# 4. 평가, 예측
# loss = model.evaluate(x, y)
def evaluate(model, criterion, x, y):
    model.eval()  # 훈련없이 평가만 하려고 함(평가모드)

    with torch.no_grad():  # grad 갱신하지 않겠음 / 1번만 돌리면 됨
        predict = model(x)
        loss2 = criterion(predict, y)
    return loss2.item()


loss2 = evaluate(model, criterion, x, y)
print('최종 loss: ', loss2)

# result = model.predict([4])
result = model(torch.Tensor([[4]]))
print('4의 예측값 : ', result.item())
'''
optimizer -> SGD
최종 loss:  4.249397079547634e-06
4의 예측값 :  3.9958434104919434
optimizer -> Adam
최종 loss:  0.039003562182188034
4의 예측값 :  3.6047184467315674
'''