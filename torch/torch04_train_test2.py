import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array(range(100))
y = np.array(range(1, 101))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)  # 랜덤난수 고정

x_train = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

print(x_train.size())
print(y_train.size())
'''
torch.Size([70, 1])
torch.Size([70, 1])
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

# 3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)


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
epoch :  96 loss :  0.138118177652359
epoch :  97 loss :  0.11021887511014938
epoch :  98 loss :  0.15472377836704254
epoch :  99 loss :  0.13341502845287323
epoch :  100 loss :  0.08963101357221603
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
# 최종 loss:  0.13765090703964233

result = model(x_test.to(DEVICE))
print(result.cpu().detach().numpy())
'''
[[ 9.516567 ]
 [93.49989  ]
 [ 5.564411 ]
 [ 6.55245  ]
 [52.990284 ]
 [42.121853 ]
 [ 1.6878734]
 [73.739105 ]
 [88.5597   ]
 [68.79891  ]
 [26.313232 ]
 [19.39696  ]
 [27.301273 ]
 [30.26539  ]
 [66.82283  ]
 [51.014206 ]
 [80.65538  ]
 [46.07401  ]
 [39.157734 ]
 [58.918518 ]
 [50.02616  ]
 [85.59558  ]
 [94.48792  ]
 [87.57165  ]
 [16.43284  ]
 [ 4.576372 ]
 [15.444802 ]
 [34.21754  ]
 [24.337152 ]
 [25.325193 ]]
'''
print('---------------------------------------')
predict_value = torch.Tensor(([[11]])).to(DEVICE)
result = model(predict_value).to(DEVICE)

print(result.cpu().detach().numpy())
# [[12.480683]]