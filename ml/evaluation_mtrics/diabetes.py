import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Binarizer

class Solution:
    def __init__(self):
        self.df = pd.read_csv('./data/diabetes.csv')

    def test(self):
        # df = pd.read_csv('./data/diabetes.csv')
        # print(df['Outcome'].value_counts())
        '''
        0    500  # Negative  
        1    268  # Positive
        '''
        # print(df.head(3))
        # print(df.info())
        '''
        <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 768 entries, 0 to 767
        Data columns (total 9 columns):
         #   Column                    Non-Null Count  Dtype  
        ---  ------                    --------------  -----  
         0   Pregnancies               768 non-null    int64  
         1   Glucose                   768 non-null    int64  
         2   BloodPressure             768 non-null    int64  
         3   SkinThickness             768 non-null    int64  
         4   Insulin                   768 non-null    int64  
         5   BMI                       768 non-null    float64
         6   DiabetesPedigreeFunction  768 non-null    float64
         7   Age                       768 non-null    int64  
         8   Outcome                   768 non-null    int64  
        dtypes: float64(2), int64(7)
        memory usage: 54.1 KB
        None
        '''
    def train_test(self):
        df = self.df
        x = df.iloc[:, :-1]
        y = df.iloc[:, -1]  # outcome 컬럼으로 레이블 값만

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66, stratify=y)

        return x_train, x_test, y_train, y_test

    def model(self):
        x_train, x_test, y_train, y_test = self.train_test()
        lr_clf = LogisticRegression(solver='liblinear')
        lr_clf.fit(x_train, y_train)
        pred = lr_clf.predict(x_test)
        pred_proba = lr_clf.predict_proba(x_test)[:, 1]
        g = self.get_clf_eval(y_test, pred, pred_proba)
        p = self.precision_recall_curve_plot(y_test, pred_proba)
        print(g)
        print(p)
        '''
        오차 행렬
        [[89 11]
         [24 30]]
        정확도: 0.7727, 정밀도: 0.7317, 재현율: 0.5556, F1: 0.6316, AUC:0.8620
        '''

    def scaler(self):
        df = self.mean_zero()
        # print(df.describe())
        x = df.iloc[:, :-1]
        y = df.iloc[:, -1]  # outcome 컬럼으로 레이블 값만

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=66, stratify=y)

        return x_train, x_test, y_train, y_test

    def model2(self):
        x_train, x_test, y_train, y_test = self.scaler()

        lr_clf = LogisticRegression()
        lr_clf.fit(x_train, y_train)
        pred = lr_clf.predict(x_test)
        pred_proba = lr_clf.predict_proba(x_test)[:, 1]
        #g = self.get_clf_eval(y_test, pred, pred_proba)
        # p = self.precision_recall_curve_plot(y_test, pred_proba)
        # print(g)
        # print(p)
        '''
        오차 행렬
        [[87 13]
         [21 33]]
        정확도: 0.7792, 정밀도: 0.7174, 재현율: 0.6111, F1: 0.6600, AUC:0.8702
        '''

        '''threshold = [0.3, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.50]
        pred_proba = lr_clf.predict_proba(x_test)
        self.get_eval_by_threshold(y_test, pred_proba[:,1].reshape(-1,1), threshold)'''
        '''
        임계값:  0.3
        오차 행렬
        [[69 31]
         [ 6 48]]
        정확도: 0.7597, 정밀도: 0.6076, 재현율: 0.8889, F1: 0.7218, AUC:0.8702
        임계값:  0.33
        오차 행렬
        [[73 27]
         [ 9 45]]
        정확도: 0.7662, 정밀도: 0.6250, 재현율: 0.8333, F1: 0.7143, AUC:0.8702
        임계값:  0.36
        오차 행렬
        [[77 23]
         [11 43]]
        정확도: 0.7792, 정밀도: 0.6515, 재현율: 0.7963, F1: 0.7167, AUC:0.8702
        임계값:  0.39
        오차 행렬
        [[80 20]
         [14 40]]
        정확도: 0.7792, 정밀도: 0.6667, 재현율: 0.7407, F1: 0.7018, AUC:0.8702
        임계값:  0.42
        오차 행렬
        [[81 19]
         [16 38]]
        정확도: 0.7727, 정밀도: 0.6667, 재현율: 0.7037, F1: 0.6847, AUC:0.8702
        임계값:  0.45
        오차 행렬
        [[82 18]
         [18 36]]
        정확도: 0.7662, 정밀도: 0.6667, 재현율: 0.6667, F1: 0.6667, AUC:0.8702
        임계값:  0.48
        오차 행렬
        [[85 15]
         [20 34]]
        정확도: 0.7727, 정밀도: 0.6939, 재현율: 0.6296, F1: 0.6602, AUC:0.8702
        임계값:  0.5
        오차 행렬
        [[87 13]
         [21 33]]
        정확도: 0.7792, 정밀도: 0.7174, 재현율: 0.6111, F1: 0.6600, AUC:0.8702
        '''

        # 임계값을 0.48로 설정한 Binarizer 생성
        binarizer = Binarizer(threshold=0.48)
        pred_th_048 = binarizer.fit_transform(pred_proba.reshape(-1, 1))
        self.get_clf_eval(y_test, pred_th_048, pred_proba)
        '''
        오차 행렬
        [[85 15]
         [20 34]]
        정확도: 0.7727, 정밀도: 0.6939, 재현율: 0.6296, F1: 0.6602, AUC:0.8702
        '''

    def mean_zero(self):
        df = self.df
        # print(df.describe())
        plt.hist(df['Glucose'], bins=100) # 0값이 일정 수준 존재
        # plt.show()
        # plt.savefig('./save/glucose_hist.png')

        # min의 값이 0인 컬럼
        #print(df.columns)
        zero_feature = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

        # 전체 데이터 건수
        total_count = df['Glucose'].count()

        #피처별로 반복 하면서 데이터 값이 0인 데이터 건수 추출하고, 퍼센트 계산
        for feature in zero_feature:
            zero_count = df[df[feature] == 0][feature].count()
            #print('{0} 0 건수는  {1}, 퍼센트는 {2: 2f} %'.format(feature, zero_count, 100*zero_count/total_count))
        '''
        Glucose 0 건수는  5, 퍼센트는  0.651042 %
        BloodPressure 0 건수는  35, 퍼센트는  4.557292 %
        SkinThickness 0 건수는  227, 퍼센트는  29.557292 %
        Insulin 0 건수는  374, 퍼센트는  48.697917 %
        BMI 0 건수는  11, 퍼센트는  1.432292 %
        '''

        # zero_features 리스트 내부에 저장된 개별 피처들에 대해서 0값을 평균 값으로 대체
        mean_zero_features = df[zero_feature].mean()
        df[zero_feature] = df[zero_feature].replace(0, mean_zero_features)
        return df

    def get_eval_by_threshold(self, y_test, pred_proba, threshold):
        # thresholds 리스트 객체내의 값을 차례로 iteration하면서 Evaluation 수행
        for custom_threshold in threshold:
            binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba)
            custom_predict = binarizer.transform(pred_proba)
            print('임계값: ', custom_threshold)
            self.get_clf_eval(y_test, custom_predict, pred_proba)

    def get_clf_eval(self, y_test, pred=None, pred_proba=None):
        confusion = confusion_matrix(y_test, pred)
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        roc_auc = roc_auc_score(y_test, pred_proba)
        print('오차 행렬')
        print(confusion)
        print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))

    def precision_recall_curve_plot(self, y_test=None, pred_proba_c1=None):
        # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출
        precision, recalls, threshold = precision_recall_curve(y_test, pred_proba_c1)

        # x축을 threshold 값으로, y축은 정밀도, 재현율 값으로 각각 plot 수행, 정밀도는 점선으로 표시
        plt.figure(figsize=(8,6))
        threshold_boundary = threshold.shape[0]
        plt.plot(threshold, precision[0:threshold_boundary], linestyle='--', label='precision')
        plt.plot(threshold, recalls[0:threshold_boundary], label='recall')

        # threshold 값 x축의 scale을 0.1 단위로 변경
        start, end = plt.xlim()
        plt.xticks(np.round(np.arange(start, end, 0.1), 2))

        #x축, y축 label과 legend, 그리고 grid 설정
        plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
        plt.legend(); plt.grid()
        #plt.show()
        #plt.savefig('./save/precision_recall.png')

if __name__ == '__main__':
    s = Solution()
    s.model2()
    # s.scaler()
    # s.mean_zero()
