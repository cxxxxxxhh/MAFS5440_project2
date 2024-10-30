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

data = pd.read_parquet('/home/cheam/fintech_pro2/datashare/GKX_bottom20.parquet').set_index(['DATE','permno'])
year = data.index.get_level_values(0).year.unique().tolist()

y_pred = []
def R_oos(actual, predicted):
    actual, predicted = np.array(actual).flatten(), np.array(predicted).flatten()
    return 1 - (np.dot((actual-predicted),(actual-predicted)))/(np.dot(actual,actual))


def cal_monthly_R2(pred, true):
    merged = pd.merge(true, pred, left_index=True, right_index=True)
    monthly_r_squared = merged.groupby(level=0).apply(lambda group: R_oos(group['RET'], group['pred']))
    #print(monthly_r_squared)
    return monthly_r_squared.mean()


def cal_yearly_R2(pred, true):
    merged = pd.merge(true, pred, left_index=True, right_index=True)
    monthly_r_squared = merged.groupby(merged.index.get_level_values(0).year).apply(lambda group: R_oos(group['RET'], group['pred']))
    return monthly_r_squared.mean()

for i in range(len(year)-30):
    train_start_date = year[0]
    #valid_start_date = year[i]+18
    test_start_date = year[i]+30
    test_end_date = year[i]+31
    train_data = data.loc[(data.index.get_level_values(0).year >= train_start_date) & (data.index.get_level_values(0).year < test_start_date)]
    #valid_data = data.loc[(data.index.get_level_values(0).year >= valid_start_date) & (data.index.get_level_values(0).year < test_start_date)]
    test_data = data.loc[(data.index.get_level_values(0).year >= test_start_date) & (data.index.get_level_values(0).year < test_end_date)]

    #l1_ratio = best_params['l1_ratio']

    enat = ElasticNet(alpha=1, l1_ratio=0, random_state=42)
    enat.fit(train_data.drop('RET',axis=1 ), train_data['RET'])
    name = '/home/cheam/fintech_pro2/enet_test/enet_lasso_recur_bottom_' + str(train_start_date) + '_' + str(test_start_date) + '.pkl'
    print(name)
    joblib.dump(enat, name)
    pred =enat.predict(test_data.drop('RET',axis=1))
    pred=pd.DataFrame(pred, index=test_data.index)
    pred.columns=['pred']
    y_pred.append(pred)

y_pred=pd.concat(y_pred)
y_pred.to_parquet('/home/cheam/fintech_pro2/enet_test/enat_lasso_recur_bottom_.parquet')

print(y_pred)
print("R^2",R_oos(data['RET'][y_pred.index], y_pred['pred']))
print("monthly R^2",cal_monthly_R2(y_pred, data['RET']))
print("yearly R^2",cal_yearly_R2(y_pred, data['RET']))






