import joblib
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
import optuna
import operator
#from xgboost import XGBRegressor

from pandas import read_parquet
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

import pickle

#data = pd.read_parquet('/home/cheam/fintech_pro2/datashare/GKX_fullset.parquet').set_index(['DATE','permno'])
#data = pd.read_parquet('/home/cheam/fintech_pro2/datashare/GKX_top20.parquet').set_index(['DATE','permno'])
data = pd.read_parquet('/home/cheam/fintech_pro2/datashare/GKX_fullset.parquet').set_index(['DATE','permno'])
year = data.index.get_level_values(0).year.unique().tolist()

def R_oos(actual, predicted):
    actual, predicted = np.array(actual).flatten(), np.array(predicted).flatten()
    return 1 - (np.dot((actual-predicted),(actual-predicted)))/(np.dot(actual,actual))


OLS_3_data = data[['RET', 'mvel1', 'bm', 'mom1m']]

# recursive 


y_pred = []
for i in range(len(year)-30): 


    train_start_date = year[0]
    valid_start_date = year[i]+18
    test_start_date = year[i]+30
    test_end_date = year[i]+31
    train_data = OLS_3_data.loc[(OLS_3_data.index.get_level_values(0).year >= train_start_date) & (OLS_3_data.index.get_level_values(0).year < test_start_date)]
    #valid_data = data.loc[(data.index.get_level_values(0).year >= valid_start_date) & (data.index.get_level_values(0).year < test_start_date)]
    test_data = OLS_3_data.loc[(OLS_3_data.index.get_level_values(0).year >= test_start_date) & (OLS_3_data.index.get_level_values(0).year < test_end_date)]
    print(train_start_date, test_start_date)

    ols_3_huber = HuberRegressor().fit(train_data[['mvel1', 'bm', 'mom1m']], train_data['RET'])#epsilon=train_data[['RET']].quantile(0.999).values[0]


    name = '/home/cheam/fintech_pro2/OLS_3_H/OLS_3_H_recur' + str(train_start_date) + '_' + str(test_start_date) + '.pkl'
    joblib.dump(ols_3_huber, name)


    pred =ols_3_huber.predict(test_data[['mvel1', 'bm', 'mom1m']])
    pred=pd.DataFrame(pred, index=test_data.index)
    pred.columns=['pred']
    y_pred.append(pred)

y_pred=pd.concat(y_pred)
print(y_pred)
y_pred.to_parquet('/home/cheam/fintech_pro2/OLS_3_H/OLS_3_H_recur_y_pred.parquet')
print("R^2",R_oos(data['RET'][y_pred.index], y_pred['pred']))

# direct 

# train_start_date = year[0]
# valid_start_date = year[18]
# test_start_date = year[30]
# train_data = OLS_3_data.loc[(OLS_3_data.index.get_level_values(0).year >= train_start_date) & (OLS_3_data.index.get_level_values(0).year < valid_start_date)]
# #valid_data = data.loc[(data.index.get_level_values(0).year >= valid_start_date) & (data.index.get_level_values(0).year < test_start_date)]
# test_data = OLS_3_data.loc[(OLS_3_data.index.get_level_values(0).year >= test_start_date)]
# ols_3_huber = HuberRegressor().fit(train_data[['mvel1', 'bm', 'mom1m']], train_data['RET'])#epsilon=train_data[['RET']].quantile(0.999).values[0]
# name = '/home/cheam/fintech_pro2/OLS_3_H/OLS_3_H_bottom20' + '.pkl'
# joblib.dump(ols_3_huber, name)
# pred =ols_3_huber.predict(test_data[['mvel1', 'bm', 'mom1m']])
# pred=pd.DataFrame(pred, index=test_data.index)
# pred.columns=['pred']
# score = R_oos(test_data['RET'], pred['pred'])
# print(pred)
# print("R^2",score)

# pred.to_parquet('/home/cheam/fintech_pro2/OLS_3_H/OLS_3_H_bottom20.parquet')

#