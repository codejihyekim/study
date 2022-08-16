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

    def sam(self):
        train = pd.read_csv('./data/train.csv')
        # test 데이터 불러오기
        test = pd.read_csv('./data/test.csv')

        sample_submission = pd.read_csv('./data/sample_submission.csv')

        for these in [train, test]:
            these['TypeofContact'] = these['TypeofContact'].fillna('Self Enquiry')
            these['DurationOfPitch'] = these['DurationOfPitch'].fillna(9.0)
            these['Gender'] = these['Gender'].replace('Fe Male', 'Female')
            these['NumberOfFollowups'] = these['NumberOfFollowups'].fillna(4.0)
            these['PreferredPropertyStar'] = these['PreferredPropertyStar'].fillna(3.0)
            these['NumberOfTrips'] = these['NumberOfTrips'].fillna(2.0)
            these['NumberOfChildrenVisiting'] = these['NumberOfChildrenVisiting'].fillna(1.0)
            these['MonthlyIncome'] = these['MonthlyIncome'].fillna(0.0)
            these['Age'] = these['Age'].fillna(35.0)
            '''ic(train['MonthlyIncome'].isna().sum())
            ic(train['MonthlyIncome'].value_counts(normalize=False))'''
        train_nona = self.handle_na(train)
        object_columns = train_nona.columns[train_nona.dtypes == 'object']

        train_enc = train_nona.copy()
        for o_col in object_columns:
            encoder = LabelEncoder()
            encoder.fit(train_enc[o_col])
            train_enc[o_col] = encoder.transform(train_enc[o_col])
        # print(train_enc)

        test = self.handle_na(test)
        for o_col in object_columns:
            encoder = LabelEncoder()
            encoder.fit(train_nona[o_col])
            test[o_col] = encoder.transform(test[o_col])
        # print(test)

        model = RandomForestClassifier()
        train = train_enc.drop(columns=['id'])
        test = test.drop(columns=['id'])

        x_train = train.drop(columns=['ProdTaken'])
        y_train = train[['ProdTaken']]

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
    s.sam()