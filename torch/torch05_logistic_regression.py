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
'''
StandardScaler은 각 열의 feature 값의 평균을 0으로 잡고, 표준편차를 1로 간주하여 정규화시키는 방법
각 데이터가 평균에서 몇 표준편차만큼 떨어져있는지를 기준으로 삼게 된다. 
데이터의 특징을 모르는 경우 가장 무난한 종류의 정규화 중 하나이다.
스케일링을 할 때 꼭 사용하는게 fit_transform(), fit(), transform() 메서드이다. 
fit_transform()은 말 그대로 fit()과 transform()을 한번에 처리할 수 있게 하는 메서드인데 
조심해야 하는 것은 테스트 데이터는 fit_transform()메서드를 쓰면 안된다. 
fit()은 데이터를 학습시키는 메서드이고 transform()은 실제로 학습시킨 것을 적용하는 메서드이다.
testset에도 fit을 해버리면 scalaer가 기존에 학습 데이터에 fit한 기준을 다 무시하고 
테스트 데이터에 새로운 mean,variance값을 얻으면서 테스트 데이터까지 학습해버린다. 
'''

print(x_train.shape, y_train.shape)
print(type(x_train), type(y_train))
'''
(398, 30) torch.Size([398, 1])
<class 'numpy.ndarray'> <class 'torch.Tensor'>
'''

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
# 2. 모델
model = nn.Sequential(
    nn.Linear(30, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),  # 1 & sigmoid 한 세트
    nn.Sigmoid()
).to(DEVICE)
'''
sigmoid 함수는 s자와 유사한 완만한 커브 형태를 보이는 함수이다. 
모든 실수 입력 값을 0보다 크고 1보다 작은 미분 가능한 수로 변환하는 특징 
sigmoid 함수는 마이너스 값을 0에 가깝게 표현하기 때문에 
입력값이 최종 계층에서 미치는 영향이 적어지는 Vanishing gradient problem이 발생 
'''

# 3. 컴파일, 훈련
criterion = nn.BCELoss()  # Binary CrossEntropy
'''
마지막 레이어가 노드수가 1개일 경우
BCELoss 함수를 사용하려면 마지막 레이어를 sigmoid함수를 적용시켜줘야한다.
'''
optimizer = optim.Adam(model.parameters(), lr=0.1)
'''
model의 parameters를 갱신하는것이 Gradient Descent에서 weight와 bias를 
갱신하는것과 같은 의미. 그래서 optim.Adam(model.parameters())해준다.
'''


def train(model, criterion, optimizer, x_train, y_train):
    model.train()  # model.eval()은 역전파 갱신이 안됨. model.train()도 있지만
    # 항상 default처럼 있기때문에 쓰지않아도 괜찮다.

    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)  # 여기까지가 순전파

    loss.backward()  # 역전파
    optimizer.step()
    return loss.item()


epoch = 0
while True:
    epoch += 1
    early_stopping = 0
    loss = train(model, criterion, optimizer, x_train, y_train)
    print(f'epoch: {epoch}, loss:{loss:.8f}')

    if epoch == 1200: break
    # if :
    #     early_stopping = 0

    # else:
    #     early_stopping += 1

    # if early_stopping == 20:
    #     break
'''
epoch: 1195, loss:0.00000021
epoch: 1196, loss:0.00000021
epoch: 1197, loss:0.00000021
epoch: 1198, loss:0.00000021
epoch: 1199, loss:0.00000021
epoch: 1200, loss:0.00000021
'''
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
# loss: 0.7479291558265686

y_predict = (model(x_test) >= 0.5).float()
# argmax가 이전에 0과1로 바꿔줬던것처럼 여기선 이런 방식으로 바꿔본다.

score = (y_predict == y_test).float().mean()
# y_predict와 y_test가 같다면 True or False로 반환되고 거기에 float해서 0과1로 바꿔준 후 평균을 내면 그게 acc다.
print(f'accyracy : {score:.4f}')
# accyracy : 0.9766

score2 = accuracy_score(y_test.cpu().numpy(), y_predict.cpu().numpy())
print(f'accuracy2 : {score2:.4f}')
# accuracy2 : 0.9766

'''
R2 회귀 점수 함수 
결정계수는 상관계수와 달리 변수간 영향을 주는 정도 또는 인과관계의 정도를 정량화해서 나타낸 수치 
따라서 결정계수는 상관분석이 아닌 회귀 분석에서 사용하는 수치라고 할 수 있다. 
결정 계수를 나타내는 R2 score는 회귀 모델의 성능에 대한 평가 지표이다. 
결정 계수가 높을수록 독립 변수가 종속 변수를 잘 설명한다는 뜻 
적합도 평가를 위한 결정계수의 R2 score는 0~1 사이의 범위를 가지고 1에 가까울수록 
해당 선형 회귀 모델이 해당 데이터에 대한 높은 연관성을 가지고 있다고 해석 
'''