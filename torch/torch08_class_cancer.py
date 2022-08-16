from turtle import forward
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7, shuffle=True, random_state=66)

x_train = torch.FloatTensor(x_train)  # .to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test)  # .to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)


class Yeram_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        # super().__init__(),
        super(Yeram_Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x


model = Yeram_Model(30, 1).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)


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
    print(f'epoch: {epoch}, loss:{loss:.8f}')

    if epoch == 1200:
        break
    # if :
    #     early_stopping = 0

    # else:
    #     early_stopping += 1

    # if early_stopping == 20:
    #     break

# 4.평가, 예측
print("================== 평가 예측 ====================")


def evaluate(model, criterion, x_test, y_test):
    model.eval()

    with torch.no_grad():
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)
    return loss.item()


loss = evaluate(model, criterion, x_test, y_test)
print(f'loss: {loss}')

y_predict = (model(x_test) >= 0.5).float()

score = accuracy_score(y_test.cpu().numpy(), y_predict.cpu().numpy())
print(f'accuracy : {score:.4f}')

'''
loss: 0.8052389025688171
accuracy2 : 0.9825
'''