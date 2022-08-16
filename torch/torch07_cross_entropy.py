from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

datasets = load_iris()  # load_iris는 classification에 적합

x = datasets.data
y = datasets.target

x = torch.FloatTensor(x)
y = torch.LongTensor(y)  # y값 자체가 int값인데 FloatTensor 하면 에러가 뜬다. LongTensor

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

x_train = torch.FloatTensor(x_train)  # .to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
x_test = torch.FloatTensor(x_test)  # .to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, y_train.shape)  # (105, 4) torch.Size([105])
print(type(x_train), type(y_train))  # <class 'numpy.ndarray'> <class 'torch.Tensor'>

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

# 2. 모델
model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
    # nn.Sigmoid()      torch에서는 마지막에 sofrmax를 안한다.
).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

'''
CrossEntropyLoss는 다중 분류에 사용 
nn.LogSoftmax와 nn.NLLLoss의 연산의 조합
nn.LogSoftmax -> 신경망 말단의 결과 값들을 확률개념으로 해석하기 위한 Softmax 함수의 결과에 log 값을 취한 연산 
nn.NLLLoss -> nn.LogSoftmax의 log 결과값에 대한 교차 엔트로피 손실연산
'''


def train(model, criterion, optimizer, x_train, y_train):
    model.train()
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()


epoch = 0
while True:
    epoch += 1
    early_stopping = 0
    loss = train(model, criterion, optimizer, x_train, y_train)
    print(f'epoch: {epoch}, loss: {loss:.8f}')

    if epoch == 1200: break
    # if :
    #     early_stopping = 0

    # else:
    #     early_stopping += 1

    # if early_stopping == 20:
    #     break

# 4.평가, 예측
print("==================평가 예측=========================")


def evaluate(model, criterion, x_test, y_test):
    model.eval()
    with torch.no_grad():
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)
    return loss.item()


loss = evaluate(model, criterion, x_test, y_test)
print(f'loss: {loss}')

y_predict = torch.argmax(model(x_test), 1)
print(y_predict)
# input tensor에 있는 모든 element들 중에서 가장 큰 값을 가지는 공간의 인덱스 번호를 반환하는 함수

scores = accuracy_score(y_test.cpu().numpy(), y_predict.cpu().numpy())
print(f'accuracy: {scores:.4f}')

'''
loss: 1.407935380935669
tensor([1, 1, 1, 0, 1, 1, 0, 0, 0, 2, 2, 2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 1, 2,
        1, 2, 0, 1, 1, 2, 0, 0, 2, 0, 1, 0, 1, 0, 2, 1, 0, 2, 2, 1, 1],
       device='cuda:0')
accuracy: 0.9111
'''