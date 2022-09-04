import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


class Solution:
    def __init__(self):
        self.df = pd.read_csv('./data/train.csv')
        self.y_df = self.df['Survived']
        self.X_df = self.df.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1)

    def test(self):
        #print(self.df)
        #print(titanic_df.info())
        '''
        <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 891 entries, 0 to 890
        Data columns (total 12 columns):
         #   Column       Non-Null Count  Dtype  
        ---  ------       --------------  -----  
         0   PassengerId  891 non-null    int64  
         1   Survived     891 non-null    int64  
         2   Pclass       891 non-null    int64  
         3   Name         891 non-null    object 
         4   Sex          891 non-null    object 
         5   Age          714 non-null    float64
         6   SibSp        891 non-null    int64  
         7   Parch        891 non-null    int64  
         8   Ticket       891 non-null    object 
         9   Fare         891 non-null    float64
         10  Cabin        204 non-null    object 
         11  Embarked     889 non-null    object 
        dtypes: float64(2), int64(5), object(5)
        memory usage: 83.7+ KB
        None
        # 결측값 있는 열 -> Sex, Cabin, Embarked
        '''
        self.dt_cl()
        self.rf_cl()
        self.lr_cl()
        dt_clf = DecisionTreeClassifier(random_state=66)
        self.exec_kfold(dt_clf, folds=5)
        self.grid_cv()

    def train_test(self):
        X_df = self.X_df
        X_df = self.encode_feature(X_df)
        X_df = self.fillna(X_df)
        #print(X_df.columns)
        X_train, X_test, y_train, y_test = train_test_split(X_df, self.y_df, test_size=0.2, random_state=11)
        return X_train, X_test, y_train, y_test

    def dt_cl(self):
        X_train, X_test, y_train, y_test = self.train_test()
        dt_clf = DecisionTreeClassifier(random_state=66)
        # DecisionTreeClassifier 학습/예측/평가
        dt_clf.fit(X_train, y_train)
        dt_pred = dt_clf.predict(X_test)
        print('DecisionTreeClassifier 정확도: {0: .4f}'.format(accuracy_score(y_test, dt_pred)))
        # DecisionTreeClassifier 정확도:  0.8212
        return dt_clf

    def rf_cl(self):
        X_train, X_test, y_train, y_test = self.train_test()
        rf_clf = RandomForestClassifier(random_state=66)
        # RandomForestClassifier 학습/예측/평가
        rf_clf.fit(X_train, y_train)
        rf_pred = rf_clf.predict(X_test)
        print('RandomForestClassifier 정확도: {0: .4f}'.format(accuracy_score(y_test, rf_pred)))
        # RandomForestClassifier 정확도:  0.8324

    def lr_cl(self):
        X_train, X_test, y_train, y_test = self.train_test()
        lr_clf = LogisticRegression(solver='liblinear')
        # LogisticRegression 학습/예측/평가
        lr_clf.fit(X_train, y_train)
        lr_pred = lr_clf.predict(X_test)
        print('LogisticRegression 정확도: {0: .4f}'.format(accuracy_score(y_test, lr_pred)))
        # LogisticRegression 정확도:  0.8492

    def exec_kfold(self, clf, folds=5):
        X_df = self.X_df
        X_df = self.encode_feature(X_df)
        X_df = self.fillna(X_df)
        # 폴드 세트를 5개인 KFold 객체를 생성, 폴드 수만큼 예측걸과 저장을 위한 리스트객체 생성
        kfold = KFold(n_splits=5)
        scores = []

        # KFold 교차 검증 수행
        for iter_count, (train_index, test_index) in enumerate(kfold.split(X_df)):
            # X_df 데이터에서 교차 검증별로 학습과 검증 데이터를 가리키는 index 생성
            X_train, X_test = X_df.values[train_index], X_df.values[test_index]
            y_train, y_test = self.y_df.values[train_index], self.y_df.values[test_index]

            # classifier 학습, 예측, 정확도 계산
            clf.fit(X_train, y_train)
            predict = clf.predict(X_test)
            acc = accuracy_score(y_test, predict)
            scores.append(acc)
            print('교차검증 {0} 정확도: {1: .4f}'.format(iter_count, acc))

        # 5개 fold에서의 평균 정확도 계산
        mean_score = np.mean(scores)
        print('평균 정확도: {0: .4f}'.format(mean_score))
        '''
        교차검증 0 정확도:  0.7542
        교차검증 1 정확도:  0.7753
        교차검증 2 정확도:  0.8146
        교차검증 3 정확도:  0.7865
        교차검증 4 정확도:  0.7921
        평균 정확도:  0.7845
        '''

    def grid_cv(self):
        dt_cl = DecisionTreeClassifier(random_state=66)
        X_train, X_test, y_train, y_test = self.train_test()
        parameters = {'max_depth':[2,3,5,10],
                      'min_samples_split': [2,3,5], 'min_samples_leaf': [1,5,8]}
        grid_dclf = GridSearchCV(dt_cl, param_grid=parameters, scoring='accuracy', cv=5)
        grid_dclf.fit(X_train, y_train)
        print('GridSearchCV 최적 하이퍼 파라미터: ', grid_dclf.best_params_)
        print('GridSearchCV 최고 정확도: {0: .4f}'.format(grid_dclf.best_score_))
        best_dclf = grid_dclf.best_estimator_

        # GridSearchCV의 최적 하이퍼 파라미터로 학습된 Estimator로 예측 및 평가 수행
        dpredictions = best_dclf.predict(X_test)
        acc = accuracy_score(y_test, dpredictions)
        print('테스트 세트에서의 DecisionTreeClassifier 정확도: {0: .4f}'.format(acc))
        '''
        GridSearchCV 최적 하이퍼 파라미터:  {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 5}
        GridSearchCV 최고 정확도:  0.7993
        테스트 세트에서의 DecisionTreeClassifier 정확도:  0.8659
        '''

    def age(self):
        df = self.df
        plt.figure(figsize=(10, 6))
        # x축의 값을 순차적으로 표시하기 위한 설정
        group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult']

        # lambda 식으로 get_category 반환값으로 지정
        df['Age_cat'] = df['Age'].apply(lambda x: self.get_category(x))
        #print(df)
        sns.barplot(x='Age_cat', y='Survived', hue='Sex', data=df, order=group_names)
        plt.show()
        #plt.savefig('./save/age.png')
        # 남성보다 여성의 생존비율이 높고, 여성중 5~12 살의 child 연령이 제일 생존 비율이 낮다.
        df.drop('Age_cat', axis=1, inplace=True)

    @staticmethod
    def get_category(age):
        cat = ''
        if age <= -1: cat = 'Unknown'
        elif age <= 5: cat = 'Baby'
        elif age <= 12: cat = 'Child'
        elif age <= 18: cat = 'Teenager'
        elif age <= 25: cat = 'Student'
        elif age <= 35: cat = 'Young Adult'
        elif age <= 60: cat = 'Adult'
        else: cat = 'Elderly'
        return cat

    @staticmethod
    def encode_feature(df):
        features = ['Sex', 'Cabin', 'Embarked']
        for feature in features:
            le = preprocessing.LabelEncoder()
            le = le.fit(df[feature])
            df[feature] = le.transform(df[feature])
        return df

    @staticmethod
    def fillna(df):
        df['Age'].fillna(df['Age'].mean(), inplace=True)
        df['Cabin'].fillna('N', inplace=True)
        df['Embarked'].fillna('N', inplace=True)
        df['Fare'].fillna(0, inplace=True)
        return df

    @staticmethod
    def drop_features(df):
        df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
        return df


if __name__ == '__main__':
    s = Solution()
    s.test()

