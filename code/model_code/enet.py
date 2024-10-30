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

def R_oos(actual, predicted):
    actual, predicted = np.array(actual).flatten(), np.array(predicted).flatten()
    return 1 - (np.dot((actual-predicted),(actual-predicted)))/(np.dot(actual,actual))

class EarlyStoppingCallback(object):
    """Early stopping callback for Optuna."""

    def __init__(self, early_stopping_rounds: int, direction: str = "minimize") -> None:
        self.early_stopping_rounds = early_stopping_rounds

        self._iter = 0

        if direction == "minimize":
            self._operator = operator.lt
            self._score = np.inf
        elif direction == "maximize":
            self._operator = operator.gt
            self._score = -np.inf
        else:
            ValueError(f"invalid direction: {direction}")

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Do early stopping."""
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            study.stop()


# optuna 优化函数
def objective(trial): #先定义很多超参数的范围，然后利用这些参数训练模型，得到test error最小的，假设我们是不知道test data的吧，只能输出cv的test error

    alpha = trial.suggest_float('alpha', 0.0001, 0.1)
    #l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)

    try:
        enat = ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=42)
        enat.fit(train_data.drop('RET',axis=1 ), train_data['RET'])
        pred =enat.predict(valid_data.drop('RET',axis=1))
        pred=pd.DataFrame(pred, index=valid_data.index)
        pred.columns=['pred']
        score = R_oos(valid_data['RET'], pred['pred'])
        return score
    except:
    #     #print(e)
        return -10000
    

y_pred = []
direction = "maximize"
for i in range(len(year)-30):
    train_start_date = year[0]
    valid_start_date = year[i]+18
    test_start_date = year[i]+30
    test_end_date = year[i]+31
    train_data = data.loc[(data.index.get_level_values(0).year >= train_start_date) & (data.index.get_level_values(0).year < valid_start_date)]
    train_data_all = data.loc[(data.index.get_level_values(0).year >= train_start_date) & (data.index.get_level_values(0).year < test_start_date)]
    valid_data = data.loc[(data.index.get_level_values(0).year >= valid_start_date) & (data.index.get_level_values(0).year < test_start_date)]
    test_data = data.loc[(data.index.get_level_values(0).year >= test_start_date) & (data.index.get_level_values(0).year < test_end_date)]

    absolute_path = "/home/cheam/fintech_pro2/parameter_tunning.db"
    print("this time", train_start_date, valid_start_date)
    print("调参")
    study = optuna.create_study(
    storage=f"sqlite:///{absolute_path}",
    study_name="enat_recur_bottom_" + str(train_start_date)+'_'+ str(valid_start_date),
    direction=direction,
    load_if_exists=True,
    )
    early_stopping = EarlyStoppingCallback(10, direction=direction)

    study.optimize(objective, n_jobs=1, callbacks=[early_stopping], timeout=600)
    best_params = study.best_params
    alpha = best_params['alpha']
    #l1_ratio = best_params['l1_ratio']
    print("score", study.best_value)

    enat = ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=42)
    enat.fit(train_data_all.drop('RET',axis=1 ), train_data_all['RET'])
    name = '/home/cheam/fintech_pro2/enat/enet_recur_bottom_' + str(train_start_date) + '_' + str(test_start_date) + '.pkl'
    print(name)
    joblib.dump(enat, name)
    pred =enat.predict(test_data.drop('RET',axis=1))
    pred=pd.DataFrame(pred, index=test_data.index)
    pred.columns=['pred']
    y_pred.append(pred)

y_pred=pd.concat(y_pred)
y_pred.to_parquet('/home/cheam/fintech_pro2/enat/enat_recur_bottom.parquet')

print(y_pred)
print("R^2",R_oos(data['RET'][y_pred.index], y_pred['pred']))


# top

#                        pred
# DATE       permno          
# 1987-01-31 10145  -0.050731
#            10401  -0.050731
#            10604  -0.050731
#            11308  -0.050731
#            11703  -0.050731
# ...                     ...
# 2016-12-31 21936  -0.036404
#            22111  -0.036404
#            22752  -0.036404
#            26403  -0.036404
#            38703  -0.036404

# [7200 rows x 1 columns]
# R^2 0.008950556646603491

# bottom 

# /home/cheam/fintech_pro2/enat/enet_recur_bottom_1957_2016.pkl
#                        pred
# DATE       permno          
# 1987-01-31 11308  -0.050466
#            12490  -0.051315
#            12570  -0.052164
#            13688  -0.048768
#            13856  -0.050749
# ...                     ...
# 2016-12-31 70519  -0.036876
#            76076  -0.036876
#            83443  -0.036876
#            84788  -0.036876
#            91233  -0.036876

# [7200 rows x 1 columns]
# R^2 0.002829527306647206


# fullset

#                        pred
# DATE       permno          
# 1987-01-31 10000  -0.046892
#            10001  -0.057046
#            10002  -0.051536
#            10003  -0.062066
#            10005  -0.067909
# ...                     ...
# 2016-12-31 93428  -0.036687
#            93429  -0.039397
#            93433  -0.041380
#            93434  -0.034220
#            93436  -0.036343

# [2481586 rows x 1 columns]
# R^2 0.0068296074956483155
