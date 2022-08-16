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
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1.1,1.2,1.3,1.4,1.5,
              1.6,1.5,1.4,1.3]])
y = np.array([11,12,13,14,15,16,17,18,19,20])

x = np.transpose(x) # (2,10) -> (10,2)

x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE) # (10) -> (10,1)

print(x, y)
print(x.shape, y.shape)
'''
tensor([[ 1.0000,  1.0000],
        [ 2.0000,  1.1000],
        [ 3.0000,  1.2000],
        [ 4.0000,  1.3000],
        [ 5.0000,  1.4000],
        [ 6.0000,  1.5000],
        [ 7.0000,  1.6000],
        [ 8.0000,  1.5000],
        [ 9.0000,  1.4000],
        [10.0000,  1.3000]], device='cuda:0') tensor([[11.],
        [12.],
        [13.],
        [14.],
        [15.],
        [16.],
        [17.],
        [18.],
        [19.],
        [20.]], device='cuda:0')
torch.Size([10, 2]) torch.Size([10, 1])
'''

# 모델 구성
# model = nn.Linear(1,1).to(DEVICE)
model = nn.Sequential(
    nn.Linear(2,5),
    nn.Linear(5,3),
    nn.Linear(3,4),
    nn.Linear(4,2),
    nn.Linear(2,1)
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
  model.train()
  optimizer.zero_grad()
  hypothesis = model(x)

  loss = criterion(hypothesis, y)
  # loss = nn.MSELoss()(hypothesis, y)
  # loss = F.mse_loss(hypothesis, y)

  loss.backward()
  optimizer.step()
  return loss.item()

epochs = 100
for epoch in range(1,epochs+1):
  loss = train(model, criterion, optimizer, x, y)
  print('epoch: {}, loss: {}'.format(epoch, loss))
'''
epoch: 96, loss: 0.5536531209945679
epoch: 97, loss: 0.5241054892539978
epoch: 98, loss: 0.49761906266212463
epoch: 99, loss: 0.47587233781814575
epoch: 100, loss: 0.45914730429649353
'''
print("==============================================")

# 4. 평가, 예측
def evaluate(model, criterion, x, y):
  model.eval()
  with torch.no_grad():
    predict = model(x)
    loss2 = criterion(predict, y)
  return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종loss : ', loss2)

result = model(torch.Tensor([10, 1.3]).to(DEVICE))
print('10, 1.3의 예측: ', result.item())
'''
최종loss :  0.4464586675167084
10, 1.3의 예측:  19.774883270263672
'''