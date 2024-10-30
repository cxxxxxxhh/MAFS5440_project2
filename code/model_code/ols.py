import joblib
from sklearn.linear_model import HuberRegressor
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
#data = pd.read_parquet('/home/cheam/fintech_pro2/datashare/GKX_fullset.parquet').set_index(['DATE','permno'])
data = pd.read_parquet('/home/cheam/fintech_pro2/datashare/GKX_fullset.parquet').set_index(['DATE','permno'])
#data = pd.read_parquet('/home/cheam/fintech_pro2/datashare/GKX_bottom20.parquet').set_index(['DATE','permno'])
def R_oos(actual, predicted):
    actual, predicted = np.array(actual).flatten(), np.array(predicted).flatten()
    return 1 - (np.dot((actual-predicted),(actual-predicted)))/(np.dot(actual,actual))


year = data.index.get_level_values(0).year.unique().tolist()

train_start_date = year[0]
valid_start_date = year[18]
test_start_date = year[30]
train_data = data.loc[(data.index.get_level_values(0).year >= train_start_date) & (data.index.get_level_values(0).year < test_start_date)]
#valid_data = data.loc[(data.index.get_level_values(0).year >= valid_start_date) & (data.index.get_level_values(0).year < test_start_date)]
test_data = data.loc[(data.index.get_level_values(0).year >= test_start_date)]
ols_huber = HuberRegressor().fit(train_data.drop('RET',axis =1), train_data['RET'])#epsilon=train_data[['RET']].quantile(0.999).values[0]
name = '/home/cheam/fintech_pro2/OLS_H/OLS_H_fullset' + '.pkl'
joblib.dump(ols_huber, name)
pred =ols_huber.predict(test_data.drop('RET',axis=1))
pred=pd.DataFrame(pred, index=test_data.index)
pred.columns=['pred']
print(pred)
score = R_oos(test_data['RET'], pred['pred'])
print("R^2",score)

pred.to_parquet('/home/cheam/fintech_pro2/OLS_H/OLS_H_y_pred_fullset.parquet')

# bottom
#                        pred
# DATE       permno          
# 1987-01-31 11308  -0.056925
#            12490  -0.070481
#            12570  -0.054840
#            13688  -0.055894
#            13856  -0.058783
# ...                     ...
# 2016-12-31 70519  -0.046851
#            76076  -0.040512
#            83443   0.045665
#            84788  -0.069665
#            91233  -0.064999

# [7200 rows x 1 columns]
# R^2 -0.2926521964431086

# top
#                        pred
# DATE       permno          
# 1987-01-31 10145  -0.112282
#            10401  -0.065773
#            10604  -0.019804
#            11308  -0.036184
#            11703  -0.038368
# ...                     ...
# 2016-12-31 21936  -0.086716
#            22111  -0.044512
#            22752  -0.088094
#            26403  -0.035176
#            38703  -0.010285

# [7200 rows x 1 columns]
# R^2 -0.32281070845966187

# fullset 

#                        pred
# DATE       permno          
# 1987-01-31 10000  -0.083356
#            10001  -0.072815
#            10002  -0.062980
#            10003  -0.073868
#            10005  -0.090857
# ...                     ...
# 2016-12-31 93428  -0.095366
#            93429  -0.063643
#            93433  -0.109253
#            93434  -0.063632
#            93436  -0.073515

# [2481586 rows x 1 columns]

# R^2 -0.06254179466635024