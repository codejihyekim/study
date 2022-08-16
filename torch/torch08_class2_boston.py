from pickletools import optimize
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, ' 사용DEVICE :', DEVICE)

# 1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=66)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape, y_train.shape) # (354, 13) torch.Size([354, 1])
# print(type(x_train), type(y_train)) # <class 'numpy.ndarray'> <class 'torch.Tensor'>

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)


# print(type(x_train), type(y_train))

# 2. 모델
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        # super().__init__()
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()

    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        out = self.linear4(x)
        return out


model = Model(13, 1).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.01)


def train(model, criterion, optimizer, x_train, y_train, batch_size=1):
    optimizer.zero_grad()

    hypothesis = model(x_train)

    loss = criterion(hypothesis, y_train)

    loss.backward()
    optimizer.step()
    return loss.item()


EPOCHS = 1000
for epoch in range(1, EPOCHS + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('Epoch : {:4d} / {:4d}, Loss : {:.8f}'.format(epoch, EPOCHS, loss))


# 4. 평가, 예측
# loss = model.evaluate(x, y)
def evaluate(model, criterion, x_test, y_test):
    model.eval()

    with torch.no_grad():
        predict = model(x_test)
        loss2 = criterion(predict, y_test)
    return loss2.item()


loss2 = evaluate(model, criterion, x_test, y_test)
print('최종 loss: ', loss2)

y_predict = model(x_test)

r2_scores = r2_score(y_test.cpu().numpy(), y_predict.cpu().detach().numpy())
print('R2: {:.4f}'.format(r2_scores))

'''
최종 loss:  2366.5673828125
R2: 0.8115
'''