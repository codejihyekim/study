# gpu
# criterion # 56번째 줄

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # loss 정의 방법

##### GPU #####
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch: ', torch.__version__, '사용DEVICE: ', DEVICE)

# 1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)  # GPU 사용
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

print(x, y)  # 스칼라가 3개
print(x.shape, y.shape)  # torch.Size([3]) torch.Size([3]) -> torch.Size([3, 1]) torch.Size([3, 1])

# 모델구성
model = nn.Linear(1, 1).to(DEVICE)

# 3. 컴파일, 훈련

criterion = nn.MSELoss()  # 평가지표의 표준
# 인스턴스(개체, 객체) / 클래스(대문자)

optimizer = optim.SGD(model.parameters(), lr=0.01)

print(optimizer)


def train(model, criterion, optimizer, x, y):
    # model.train()  # 훈련모드
    optimizer.zero_grad()  # 기울기 초기화

    hypothesis = model(x)  # x를 넣었을 때의 값이 hypothesis에 담김 # y = wx + b

    # loss = criterion(hypothesis, y) # 예측값과 실제값 비교 # MSE
    # loss = nn.MSELoss(hypothesis, y) # 에러
    # loss = nn.MSELoss()(hypothesis, y) # 정상작동 (방법1)
    loss = F.mse_loss(hypothesis, y)  # 정상작동 (방법2)

    # 여기까지가 순전파

    loss.backward()  # 기울기값 계산까지
    optimizer.step()  # 가중치 수정(역전파)
    return loss.item()


epochs = 100
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epoch, loss))
'''
optimizer -> SGD
epoch: 96, loss: 0.10973694920539856
epoch: 97, loss: 0.10920996963977814
epoch: 98, loss: 0.10868556797504425
epoch: 99, loss: 0.10816362500190735
epoch: 100, loss: 0.1076442152261734
'''
print("=============================================")


# 4. 평가, 예측
# loss = model.evaluate(x,y)
def evaluate(model, criterion, x, y):
    model.eval()  # 훈련없이 평가만(평가모드)
    with torch.no_grad():
        predict = model(x)
        loss2 = criterion(predict, y)
    return loss2.item()


loss2 = evaluate(model, criterion, x, y)
print('최종 loss: ', loss2)

# result = model.predict([4])
result = model(torch.Tensor([[4]]).to(DEVICE))
print('4의 예측값: ', result.item())
'''
optimizer -> SGD
최종 loss:  0.10712730139493942
4의 예측값:  3.343564987182617
'''
