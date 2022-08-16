import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# 1. 데이터
x_train = np.array([1, 2, 3, 4, 5, 6, 7])
x_test = np.array([8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7])
y_test = np.array([8, 9, 10])

x_train = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

print(x_train.size())
'''
tensor([[1.],
        [2.],
        [3.],
        [4.],
        [5.],
        [6.],
        [7.]], device='cuda:0')
torch.Size([7, 1])
'''
# 2. 모델
model = nn.Sequential(
    nn.Linear(1, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.ReLU(),
    nn.Linear(3, 4),
    nn.Linear(4, 1),
).to(DEVICE)
'''
ReLU는 0보다 작아지게 되면 0이 되는 특징을 가지고 있다.
이러한 특징에 의해서 미분 값은 0과 1만을 가지고 계속해서 작아지는 문제를 줄여주게 된다. 
하지만 입력값이 음수이면 미분 값이 0이다보니 weight를 완전히 죽여버리게 된다. 
'''
# 3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
'''
MSELoss는 예측과 타겟값의 차이를 제곱하여 평균한 값(모두 실수값으로 계산)
MSE가 크다는 것은 평균 사이의 차이가 크다는 뜻 
즉 MSE는 데이터가 평균으로부터 얼마나 떨어져있나를 보여주는 손실함수 
'''


def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()

    hyporthesis = model(x)
    loss = criterion(hyporthesis, y)

    loss.backward()
    optimizer.step()

    return loss.item()


EPOCHS = 100
for epoch in range(1, EPOCHS + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print("epoch : ", epoch, "loss : ", loss)
'''
epoch :  96 loss :  0.0004721850564237684
epoch :  97 loss :  0.00014169994392432272
epoch :  98 loss :  0.0005355303874239326
epoch :  99 loss :  0.0004728278727270663
epoch :  100 loss :  9.0441251813899
'''


# 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()  # 평가모드

    with torch.no_grad():
        predict = model(x)
        loss2 = criterion(predict, y)
    return loss2.item()


loss2 = evaluate(model, criterion, x_test, y_test)
print("최종 loss: ", loss2)
# 최종 loss:  0.0019713793881237507

result = model(x_test.to(DEVICE))
print(result.cpu().detach().numpy())
# [[7.962879][8.95596 ][9.949042]]

'''
Pytorch를 사용하다보면 Module을 통해 나온 Tensor을 후처리에 사용하거나, 계산된 loss Tensor을 logging하는 일이 많다. 
이때 result는 GPU에 올라가 있는 Tensor이기 때문에 numpy 혹은 list로의 변환이 필요하다. 
datach() -> tensor에서 이루어진 모든 연산을 추적해서 기록해놓는다. 연산 기록으로부터 분리한 tensor을 반환
cpu() -> GPU 메모리에 올려져 있는 tensor을 cpu 메모리로 복사
numpy() -> tensor를 numpy로 변환하여 반환
'''
print('---------------------------------------')
predict_value = torch.Tensor(([[11]])).to(DEVICE)
result = model(predict_value).to(DEVICE)
print(result.cpu().detach().numpy())
# [[10.942123]]