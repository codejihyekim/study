import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns

class Solution:
    def __init__(self):
        self.feature_name_df = pd.read_csv('./data/features.txt', sep='\s+', header=None, names=['column_index', 'column_name'])

    def test(self):
        #feature.txt 파일에는 피처 이름 index와 피처명이 공백으로 분리되어 있음
        feature_name_df = self.feature_name_df
        #print(feature_name_df)

        #피처명 index를 제거하고, 피처명만 리스트 객체로 생성한 뒤 샘플로 10개만 추출
        feature_name = feature_name_df.iloc[:, 1].values.tolist()
        #print('전체 피처명에서 10개만 추출', feature_name[:10])
        '''
        전체 피처명에서 10개만 추출 ['tBodyAcc-mean()-X', 'tBodyAcc-mean()-Y', 'tBodyAcc-mean()-Z', 'tBodyAcc-std()-X', 
        'tBodyAcc-std()-Y', 'tBodyAcc-std()-Z', 'tBodyAcc-mad()-X', 'tBodyAcc-mad()-Y', 'tBodyAcc-mad()-Z', 'tBodyAcc-max()-X']
        '''
        feature_dup_df = feature_name_df.groupby('column_name').count()
        #print(feature_dup_df)
        #print(feature_dup_df[feature_dup_df['column_index'] > 1].count())
        #print(feature_dup_df[feature_dup_df['column_index'] > 1].head())

        x_train, x_test, y_train, y_test = self.get_human_dataset()
        #print(x_train.shape) # (7352, 561)
        #print(x_train.info())
        '''
        <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 7352 entries, 0 to 7351
        Columns: 561 entries, tBodyAcc-mean()-X to angle(Z,gravityMean)
        dtypes: float64(561)
        memory usage: 31.5 MB
        '''
        #print(y_train['action'].value_counts())
        '''
        6    1407
        5    1374
        4    1286
        1    1226
        2    1073
        3     986
        Name: action, dtype: int64
        '''
        dt_clf = DecisionTreeClassifier(random_state=156)
        dt_clf.fit(x_train, y_train)
        pred = dt_clf.predict(x_test)
        acc = accuracy_score(y_test, pred)
        print('결정트리 예측 정확도: {0: .4f}'.format(acc))
        # 결정트리 예측 정확도:  0.8548

        # DecisionTreeClassifie의 하이퍼파라미터 추출
        print('DecisionTreeClassifier 기본 하이퍼 파라미터: \n', dt_clf.get_params())
        '''
        DecisionTreeClassifier 기본 하이퍼 파라미터: 
        {'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 
        'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 
        'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': 156, 'splitter': 'best'}
        '''

    def tree_depth(self):
        # 결정 트리의 트리 깊이가 예측 정확도에 주는 영향
        # min_samples_split는 16으로 고정, max_depth를 늘리면서 예측 성장을 측정
        x_train, x_test, y_train, y_test = self.get_human_dataset()
        dt_clf = DecisionTreeClassifier(random_state=156)
        params = {
            'max_depth': [6,8,10,12,16,20,24],
            'min_samples_split': [16]
        }
        grid_cv = GridSearchCV(dt_clf, param_grid=params, scoring='accuracy', cv=5, verbose=1)
        grid_cv.fit(x_train, y_train)
        #print('GridSearchCV 최고 평균 정확도 수치: {0: .4f}'.format(grid_cv.best_score_))
        #print('GridSearchCV 최적 하이퍼 파라미터: ', grid_cv.best_params_)
        '''
        GridSearchCV 최고 평균 정확도 수치:  0.8549
        GridSearchCV 최적 하이퍼 파라미터: {'max_depth': 8, 'min_samples_split': 16}
        # max_depth가 8일 때 5개의 폴드 세트의 최고 정확도 결과가 약 85.49로 도출 
        '''
        # cv_resutls_ 속성을 Dataframe으로 생성
        cv_result_df = pd.DataFrame(grid_cv.cv_results_)
        # max_depth 파라미터 값과 그때의 테스트 세트, 학습 데이터 세트의 정확도 수치 추출
        #print(cv_result_df[['param_max_depth', 'mean_test_score']])
        '''
          param_max_depth  mean_test_score
        0               6         0.847662
        1               8         0.854879 *
        2              10         0.852705
        3              12         0.845768
        4              16         0.847127
        5              20         0.848624
        6              24         0.848624
        '''

    def tree_depth_test(self):
        x_train, x_test, y_train, y_test = self.get_human_dataset()
        max_depth = [6,8,10,12,16,20,24]
        #max_depth 값을 변화시키면서 그때마다 학습과 테스트 세트에서의 예측 성능 측정
        for depth in max_depth:
            dt_clf = DecisionTreeClassifier(max_depth=depth, min_samples_split=16, random_state=156)
            dt_clf.fit(x_train, y_train)
            pred = dt_clf.predict(x_test)
            acc = accuracy_score(y_test, pred)
            print('max_depth = {0} 정확도: {1: .4f}'.format(depth, acc))
        '''
        max_depth = 6 정확도:  0.8551
        max_depth = 8 정확도:  0.8717 *
        max_depth = 10 정확도:  0.8599
        max_depth = 12 정확도:  0.8571
        max_depth = 16 정확도:  0.8599
        max_depth = 20 정확도:  0.8565
        max_depth = 24 정확도:  0.8565
        '''

    def final_tree_depth(self):
        x_train, x_test, y_train, y_test = self.get_human_dataset()
        dt_clf = DecisionTreeClassifier(random_state=156)
        param = {
            'max_depth': [8,12,16,20],
            'min_samples_split': [16,24]
        }
        grid_cv = GridSearchCV(dt_clf, param_grid=param, scoring='accuracy', cv=5, verbose=1)
        grid_cv.fit(x_train, y_train)
        print('GridSearchCV 최고 평균 정확도 수치: {0: .4f}'.format(grid_cv.best_score_))
        print('GridSearchCV 최적 하이퍼 파라미터: ', grid_cv.best_params_)

        best_df_clf = grid_cv.best_estimator_
        pred = best_df_clf.predict(x_test)
        acc = accuracy_score(y_test, pred)
        print('결정 트리 예측 정확도: {0: .4f}'.format(acc))

        ftr_importances_values = best_df_clf.feature_importances_
        # top 중요도로 정렬을 쉽게 하고, 시본의 막대그래프로 쉽게 표현하기 위해 series 변환
        ftr_importances = pd.Series(ftr_importances_values, index=x_train.columns)
        # 중요도값 순으로 Series를 정렬
        ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
        plt.figure(figsize=(8,6))
        plt.title('Feature importances Top 20')
        sns.barplot(x=ftr_top20, y=ftr_top20.index)
        #plt.show()
        #plt.savefig('./save/depth_split_importances.png')

    def get_human_dataset(self):
        feature_name_df = self.feature_name_df

        # 중복된 피처명을 수정하는 get_new_feature_name_df()를 이용. 신규 피처명 dataframe 생성
        new_feature_name_df = self.get_new_feature_name_df(feature_name_df)

        # Dataframe에 피처명을 칼럼으로 부여하기 위해 리스트 객체로 다시 변환
        feature_name = new_feature_name_df.iloc[:, 1].values.tolist()
        #print(feature_name)

        # 학습 피처 데이터세트와 테스트 피처 데이터를 Dataframe으로 로딩, 칼럼명은 feature_name 적용
        x_train = pd.read_csv('./data/X_train.txt', sep='\s+', names=feature_name)
        x_test = pd.read_csv('./data/X_test.txt', sep='\s+', names=feature_name)

        # 학습 레이블과 테스트 레이블 데이터를 Dataframe으로 로딩하고 칼럼명은 action으로 부여
        y_train = pd.read_csv('./data/y_train.txt', sep='\s+', header=None, names=['action'])
        y_test = pd.read_csv('./data/y_test.txt', sep='\s+', header=None, names=['action'])

        # 로드된 학습/테스트용 Dataframe을 모두 반환
        return x_train, x_test, y_train, y_test

    def get_new_feature_name_df(self, old_feature_name_df):
        feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby('column_name').cumcount(), columns=['dup_cnt'])
        feature_dup_df = feature_dup_df.reset_index()
        new_feature_name_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how='outer')
        new_feature_name_df['column_name'] = new_feature_name_df[['column_name', 'dup_cnt']].apply(lambda x : x[0]+'_'+ str(x[1]) if x[1] >0 else x[0], axis=1)
        new_feature_name_df = new_feature_name_df.drop(['index'], axis=1)
        return new_feature_name_df

if __name__ == '__main__':
    s = Solution()
    #s.get_human_dataset()
    #s.test()
    #s.tree_depth()
    #s.tree_depth_test()
    s.final_tree_depth()