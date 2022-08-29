import os

import numpy as np
import pandas as pd
import seaborn as sb
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

plt.rcParams['figure.figsize']=(10,10)
plt.rcParams['font.family']='AppleGothic'

warnings.filterwarnings(action='ignore')

def read_csv_by_dir(path, index_col=None):
    df_raw = pd.DataFrame()
    for files in os.listdir(path): # os.listdir()는 지정한 디렉토리 내의 모든 파일과 디렉토리의 리스트를 리턴한다.
        if files.endswith('.csv'):
            df = pd.read_csv('/'.join([path,files]),
                            index_col=index_col)
        df_raw = pd.concat((df_raw,df),axis=0)
    return df_raw

path = './data'
_df_rf_raw = read_csv_by_dir('/'.join([path,'rf_data']),
                            index_col=0)

_df_water_raw = read_csv_by_dir('/'.join([path,'water_data']),
                               index_col=0)

_submission_raw = pd.read_csv('/'.join([path,'sample_submission.csv']),
                             index_col=0)
# raw_data 보존하기
df_rf=_df_rf_raw.copy()
df_rf.name = "rain_data"

df_water=_df_water_raw.copy()
df_water.name = "water_data"

submission=_submission_raw.copy()
submission.name = "submission"

def index_to_datetime(df,format):
    df.index = pd.to_datetime(df.index,
                              format=format)
    return df

df_rf=index_to_datetime(df=df_rf,format='%Y-%m-%d %H:%M')
df_water=index_to_datetime(df=df_water,format='%Y-%m-%d %H:%M')
submission=index_to_datetime(df=submission,format='%Y-%m-%d %H:%M')

df_rf.sort_index(inplace=True)
df_water.sort_index(inplace=True)
submission.sort_index(inplace=True)

# 데이터 시간대 확인하기
def check_datetime(df):
    print(df.name)
    print(df.select_dtypes('datetime64[ns]').head(1).index[0])
    print(df.select_dtypes('datetime64[ns]').tail(1).index[0])
    return None

check_datetime(df_rf)
check_datetime(df_water)
check_datetime(submission)

# data target 분리하기
target = df_water.loc[:,submission.columns]
data = pd.concat((df_rf,df_water.drop(submission.columns,axis=1)),axis=1)

# data와 target 하나 밀어주기 (과거데이터를 사용해야 함으로)
_target = target.reset_index(drop=True)
_data = data.reset_index(drop=True)

_data.index += 1

tot=pd.concat((_data,_target),axis=1)
tot=tot.sort_index()

tot=tot.iloc[1:-1]
#print(tot.value_counts())
target = tot.loc[:,submission.columns]
data = tot.drop(submission.columns,axis=1)

print(data.corr(method='pearson'))
plt.rcParams["figure.figsize"] = (17,17)
sb.heatmap(data.corr(), annot=True, cmap='Greens', vmin=-1, vmax=1)
plt.show()

train_target=target.iloc[:-len(submission),:]
test_target=target.iloc[-len(submission):,:]

train_data=data.iloc[:-len(submission),:]
test_data=data.iloc[-len(submission):,:]

train_target.fillna(train_target.mean(),inplace=True)
test_target.fillna(train_target.mean(),inplace=True)
train_data.fillna(train_data.mean(),inplace=True)
test_data.fillna(train_data.mean(),inplace=True)

print('--data--')
print(train_data.shape)
print(test_data.shape)
print('--target--')
print(train_target.shape)
print(test_target.shape)

from sklearn.model_selection import KFold
kfold = KFold(n_splits=2, shuffle=True)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_jobs=-1)

params = {
    "n_estimators" : (100, 150, 200)
}

from sklearn.model_selection import GridSearchCV
grid_cv = GridSearchCV(rf,
                       param_grid=params,
                       cv = kfold,
                       n_jobs=-1)

grid_cv.fit(train_data,train_target)

grid_cv.best_estimator_

model=grid_cv.best_estimator_
model.fit(train_data,train_target)
y_pred=model.predict(test_data)

_submission_raw.iloc[:,:] = y_pred
_submission_raw.to_csv('./save/ans.csv')