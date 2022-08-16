import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # loss 정의 방법

#### GPU ####
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch: ', torch.__version__, '사용DEVICE: ', DEVICE)

# 1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

# 모델구성
# model = nn.Linear(1, 1).to(DEVICE)
model = nn.Sequential(
    nn.Linear(1, 5),
    nn.Linear(5, 3),
    nn.Linear(3, 4),
    nn.Linear(4, 2),
    nn.Linear(2, 1),
).to(DEVICE)

print(model)
# forward() 함수에서 구현될 순전파를 Layer 형태로 보다 가독성이 뛰어나게 코드를 작성할 수 있다.
# Layer가 복잡할수록 nn.Sequential은 효과가 좋다.
# Sequnetial에 쓰인 순서대로 작업된다.

# 3. 컴파일, 훈련
criterion = nn.MSELoss()  # 평가지표의 표준
# optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.01)


def train(model, criterion, optimizer, x, y):
    model.train()
    optimizer.zero_grad()  # 기울기 초기화
    hypothesis = model(x)
    # loss = criterion(hypothesis, y)
    # loss = nn.MSELoss()(hypothesis, y)
    loss = F.mse_loss(hypothesis, y)

    loss.backward()
    optimizer.step()
    return loss.item()


epochs = 100
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epoch, loss))
'''
epoch: 96, loss: 0.13408544659614563
epoch: 97, loss: 0.1324235498905182
epoch: 98, loss: 0.1307745724916458
epoch: 99, loss: 0.12913858890533447
epoch: 100, loss: 0.12751559913158417
'''
print("===============================================")


# 4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        predict = model(x)
        loss2 = criterion(predict, y)
    return loss2.item()


loss2 = evaluate(model, criterion, x, y)
print('최종 loss: ', loss2)

result = model(torch.Tensor([3]).to(DEVICE))
print('3의 예측값: ', result.item())
'''
최종 loss:  0.12590548396110535
3의 예측값:  2.6850593090057373
'''
