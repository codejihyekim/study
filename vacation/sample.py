import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from icecream import ic
import seaborn as sb
import sklearn.metrics as metrics

class Solution:
    def __init__(self):
        pass

    def proprecessing(self):
        #train 데이터 불러오기
        train = pd.read_csv('./data/train.csv')

        #test 데이터 불러오기
        test = pd.read_csv('./data/test.csv')

        # sample_submissin 불러오기
        sample_submission = pd.read_csv('./data/sample_submission.csv')
        # print(train.info())

        # plt.hist(train.ProdTaken)
        # plt.show()
        # 여행 상품을 신청하지 않은 사람들이(0) 신청한 사람들(1)에 비해 약 3배 가량 많은 모습을 볼 수 있다.

        # print(train.isna().sum())

        train_nona = self.handle_na(train)
        #train_nona.to_csv('./save/train_a.csv', index=False)

        object_columns = train_nona.columns[train_nona.dtypes == 'object']
        # print('object 칼럼은 다음과 같습니다:', list(object_columns))

        encoder = LabelEncoder()
        encoder.fit(train_nona['TypeofContact'])
        #print(encoder.transform(train_nona['TypeofContact'])) # [0 1 0 ... 0 1 0]

        train_enc = train_nona.copy()
        for o_col in object_columns:
            encoder = LabelEncoder()
            encoder.fit(train_enc[o_col])
            train_enc[o_col] = encoder.transform(train_enc[o_col])
        #print(train_enc)

        test = self.handle_na(test)
        for o_col in object_columns:
            encoder = LabelEncoder()
            encoder.fit(train_nona[o_col])
            test[o_col] = encoder.transform(test[o_col])
        #print(test)

        '''print(train_enc.corr(method='pearson'))
        plt.rcParams["figure.figsize"] = (20,20)
        sb.heatmap(train_enc.corr(), annot=True, cmap='Greens', vmin=-1, vmax=1)
        plt.show()'''

        model = RandomForestClassifier()
        train = train_enc.drop(columns=['id'])
        test = test.drop(columns=['id'])

        x_train = train.drop(columns=['ProdTaken'])
        y_train = train[['ProdTaken']]

        print(y_train)

        model.fit(x_train, y_train)

        print(model.score(x_train, y_train))

        prediction = model.predict(test)
        print('----------예측된 데이터의 상위 10개의 값 확인--------------')
        print(prediction)

        sample_submission['ProdTaken'] = prediction
        print(sample_submission.head())
        sample_submission.to_csv('./save/submission.csv', index=False)

    def handle_na(self, data):
        temp = data.copy()
        for col, dtype in temp.dtypes.items():
            value = ''
            if dtype == 'object':
                value = 'Unknown'
            elif dtype == int or dtype == float:
                value = 0
            temp.loc[:,col] = temp[col].fillna(value)
        return temp

if __name__ == '__main__':
    s = Solution()
    s.proprecessing()


