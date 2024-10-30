# %%
import joblib
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
#from xgboost import XGBRegressor

from pandas import read_parquet
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import pickle
import seaborn as sns 

# %%
def R_oos(actual, predicted):
    actual, predicted = np.array(actual).flatten(), np.array(predicted).flatten()
    return 1 - (np.dot((actual-predicted),(actual-predicted)))/(np.dot(actual,actual))


def cal_monthly_R2(pred, true):
    merged = pd.merge(true, pred, left_index=True, right_index=True)
    monthly_r_squared = merged.groupby(level=0).apply(lambda group: R_oos(group['RET'], group['pred']))
    #print(monthly_r_squared)
    return monthly_r_squared


def cal_yearly_R2(pred, true):
    merged = pd.merge(true, pred, left_index=True, right_index=True)
    monthly_r_squared = merged.groupby(merged.index.get_level_values(0).year).apply(lambda group: R_oos(group['RET'], group['pred']))
    #print(monthly_r_squared)
    return monthly_r_squared

# %%
true = pd.read_parquet('/home/cheam/fintech_pro2/datashare/GKX_no_interaction.parquet').set_index(['DATE','permno'])
true

# %% [markdown]
# ## OLS3

# %%
pred_full = pd.read_parquet('/home/cheam/fintech_pro2/OLS_3_H/OLS_3_H_fullset.parquet')
pred_top = pd.read_parquet('/home/cheam/fintech_pro2/OLS_3_H/OLS_3_H_top20.parquet')
pred_bottom = pd.read_parquet('/home/cheam/fintech_pro2/OLS_3_H/OLS_3_H_bottom20.parquet')

# %%
print("fullset",R_oos(true['RET'][pred_full.index], pred_full['pred']))
print("top",R_oos(true['RET'][pred_top.index], pred_top['pred']))
print("bototm",R_oos(true['RET'][pred_bottom.index], pred_bottom['pred']))
month_full = cal_monthly_R2(pred_full, true)
print("end")
month_top = cal_monthly_R2(pred_top, true)
print("end")
month_bottom = cal_monthly_R2(pred_bottom, true)
print("end")
year_full = cal_yearly_R2(pred_full, true)
print("end")
year_top = cal_yearly_R2(pred_top, true)
print("end")
year_bottom = cal_yearly_R2(pred_bottom, true)
print("end")
print("month_full_R2:", month_full.mean())
print("month_top_R2:", month_top.mean())
print("month_bottom_R2:", month_bottom.mean())
print("year_full_R2:", year_full.mean())
print("year_top_R2:", year_top.mean())
print("year_bottom_R2:", year_bottom.mean())

# %%
month_full = cal_monthly_R2(pred_full, true)
month_full = pd.DataFrame(month_full,columns=['R2'])
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(month_full.index, month_full['R2'], label='R2', color='blue')
plt.xticks(month_full.index[::30], rotation=45)
plt.title('Monthly R2 of OLS3')
plt.xlabel('Date')
plt.ylabel('R2')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# %%
year_full = cal_yearly_R2(pred_full, true)
year_full = pd.DataFrame(year_full,columns=['R2'])
import matplotlib.pyplot as plt
sns.set()
plt.figure(figsize=(12, 6))
plt.plot(year_full.index, year_full['R2'], label='R2', color='blue')
plt.xticks(year_full.index[::5], rotation=45)
plt.xlabel('Date')
plt.ylabel('R2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
year_full = cal_yearly_R2(pred_full, true)
year_full = pd.DataFrame(year_full,columns=['R2'])
sns.set()
plt.figure(figsize=(10, 5))
plt.plot(year_full.index, year_full['R2'], marker='o', linestyle='-', color='royalblue', linewidth=2, markersize=6)
plt.xlabel("Year")
plt.ylabel("Value")
# plt.title("Tree Depth vs Year for XGBoost Model Complexity")
plt.grid(True)
plt.xticks(rotation=0)
plt.show()

# %% [markdown]
# # OLS

# %%
pred_full = pd.read_parquet('/home/cheam/fintech_pro2/OLS_H/OLS_H_y_pred_full.parquet')
pred_top = pd.read_parquet('/home/cheam/fintech_pro2/OLS_H/OLS_H_y_pred_top.parquet')
pred_bottom = pd.read_parquet('/home/cheam/fintech_pro2/OLS_H/OLS_H_y_pred_bottom.parquet')

# %%
print("fullset",R_oos(true['RET'], pred_full['pred']))
print("top",R_oos(true['RET'][pred_top.index], pred_top['pred']))
print("bototm",R_oos(true['RET'][pred_top.index], pred_bottom['pred']))
month_full = cal_monthly_R2(pred_full, true)
month_top = cal_monthly_R2(pred_top, true)
month_bottom = cal_monthly_R2(pred_bottom, true)
year_full = cal_yearly_R2(pred_full, true)
year_top = cal_yearly_R2(pred_top, true)
year_bottom = cal_yearly_R2(pred_bottom, true)
print("month_full_R2:", month_full.mean())
print("month_top_R2:", month_top.mean())
print("month_bottom_R2:", month_bottom.mean())
print("year_full_R2:", year_full.mean())
print("year_top_R2:", year_top.mean())
print("year_bottom_R2:", year_bottom.mean())

# %%
#year_full = cal_yearly_R2(pred_full, true)
year_full = pd.DataFrame(year_full,columns=['R2'])
plt.figure(figsize=(12, 6))
plt.plot(year_full.index, year_full['R2'], label='R2', color='blue')
plt.xticks(year_full.index[::5], rotation=45)
plt.title('Yearly R2 of OLS')
plt.xlabel('Date')
plt.ylabel('R2')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# %% [markdown]
# # Enet

# %%
pred_recur_full  = pd.read_parquet('/home/cheam/fintech_pro2/enat/enat_recur_fullset.parquet')
pred_recur_top  = pd.read_parquet('/home/cheam/fintech_pro2/enat/enat_recur_top.parquet')
pred_recur_bottom  = pd.read_parquet('/home/cheam/fintech_pro2/enat/enat_recur_bottom.parquet')
pred_lasso_recur_full  = pd.read_parquet('/home/cheam/fintech_pro2/enet_test/enat_lasso_recur_.parquet')
pred_lasso_recur_top  = pd.read_parquet('/home/cheam/fintech_pro2/enet_test/enat_lasso_recur_top_.parquet')
pred_lasso_recur_bottom  = pd.read_parquet('/home/cheam/fintech_pro2/enet_test/enat_lasso_recur_bottom_.parquet')

# %%
print("fullset",R_oos(true['RET'][pred_recur_full.index], pred_recur_full['pred']))
print("top",R_oos(true['RET'][pred_top.index], pred_recur_top['pred']))
print("bototm",R_oos(true['RET'][pred_bottom.index], pred_recur_bottom['pred']))
month_full = cal_monthly_R2(pred_recur_full, true)
print("end")
month_top = cal_monthly_R2(pred_recur_top, true)
print("end")
month_bottom = cal_monthly_R2(pred_recur_bottom, true)
print("end")
year_full = cal_yearly_R2(pred_recur_full, true)
print("end")
year_top = cal_yearly_R2(pred_recur_top, true)
print("end")
year_bottom = cal_yearly_R2(pred_recur_bottom, true)
print("end")
print("month_full_R2:", month_full.mean())
print("month_top_R2:", month_top.mean())
print("month_bottom_R2:", month_bottom.mean())
print("year_full_R2:", year_full.mean())
print("year_top_R2:", year_top.mean())
print("year_bottom_R2:", year_bottom.mean())

# %%
print("fullset",R_oos(true['RET'][pred_lasso_recur_full.index], pred_lasso_recur_full['pred']))
print("top",R_oos(true['RET'][pred_top.index], pred_lasso_recur_top['pred']))
print("bototm",R_oos(true['RET'][pred_top.index], pred_lasso_recur_bottom['pred']))
month_full_lasso = cal_monthly_R2(pred_lasso_recur_full, true)
month_top_lasso = cal_monthly_R2(pred_lasso_recur_top, true)
month_bottom_lasso = cal_monthly_R2(pred_lasso_recur_bottom, true)
year_full_lasso = cal_yearly_R2(pred_lasso_recur_full, true)
year_top_lasso = cal_yearly_R2(pred_lasso_recur_top, true)
year_bottom_lasso = cal_yearly_R2(pred_lasso_recur_bottom, true)
print("month_full_R2:", month_full_lasso.mean())
print("month_top_R2:", month_top_lasso.mean())
print("month_bottom_R2:", month_bottom_lasso.mean())
print("year_full_R2:", year_full_lasso.mean())
print("year_top_R2:", year_top_lasso.mean())
print("year_bottom_R2:", year_bottom_lasso.mean())

# %%
year_full_lasso = pd.DataFrame(year_full_lasso,columns=['Test R2 of Lasso'])
year_full = pd.DataFrame(year_full,columns=['Test R2 of ENet'])

# %%
year_r = pd.merge(year_full, year_full_lasso, left_index=True, right_index=True)
year_r

# %%
year_r['Validation R2 of ENet'] = [0.330731245,0.325553874
,0.313803866
,0.337348385
,0.344390011
,0.338644365
,0.345382567
,0.339734257
,0.329973691
,0.300608427
,0.252731191
,0.216093261
,0.178015209
,0.122295434
,0.115085645
,0.089071078
,0.050631103
,0.018874785
,0.00094531
,0.017779523
,0.018709069
,0.025296736
,0.03460895
,-0.002702659
,-0.018853166
,-0.021175736
,-0.047017232
,-0.065153717
,-0.079257849
,-0.057554739]
year_r

# %%

sns.set()
plt.figure(figsize=(10, 5))
#plt.plot(years, depths, marker='o', linestyle='-', color='royalblue', linewidth=2, markersize=6)
plt.plot(year_r.index, year_r['Test R2 of ENet'], label='Test R2 of ENet', color='blue',marker='o', linestyle='-', linewidth=2, markersize=6)
plt.plot(year_r.index, year_r['Test R2 of Lasso'], label='Test R2 of Lasso', color='orange',marker='o', linestyle='-', linewidth=2, markersize=6)
plt.plot(year_r.index, year_r['Validation R2 of ENet'], label='Validation R2 of ENet', color='green',marker='o', linestyle='-', linewidth=2, markersize=6)
plt.xticks(year_r.index[::5], rotation=45)
# plt.title('R2 of ENet and Lasso')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
#month_full = cal_monthly_R2(pred_full, true)
month_full = pd.DataFrame(month_full,columns=['R2'])
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(month_full.index, month_full['R2'], label='R2', color='blue')
plt.xticks(month_full.index[::30], rotation=45)
plt.title('Monthly R2 of OLS3')
plt.xlabel('Date')
plt.ylabel('R2')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# %%
#year_full = cal_yearly_R2(pred_full, true)
year_full = pd.DataFrame(year_full,columns=['R2'])
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(year_full.index, year_full['R2'], label='R2', color='blue')
plt.xticks(year_full.index[::5], rotation=45)
plt.title('Yearly R2 of OLS3')
plt.xlabel('Date')
plt.ylabel('R2')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# %%
#month_full = cal_monthly_R2(pred_full, true)
month_full = pd.DataFrame(month_full,columns=['R2'])
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(month_full.index, month_full['R2'], label='R2', color='blue')
plt.xticks(month_full.index[::30], rotation=45)
plt.title('Monthly R2 of OLS3')
plt.xlabel('Date')
plt.ylabel('R2')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# %%
#year_full = cal_yearly_R2(pred_full, true)
year_full = pd.DataFrame(year_full,columns=['R2'])
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(year_full.index, year_full['R2'], label='R2', color='blue')
plt.xticks(year_full.index[::5], rotation=45)
plt.title('Yearly R2 of OLS3')
plt.xlabel('Date')
plt.ylabel('R2')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# %% [markdown]
# # DM-test

# %%
pred_ols3 = pd.read_parquet('/home/cheam/fintech_pro2/OLS_3_H/OLS_3_H_fullset.parquet')
pred_ols = pd.read_parquet('/home/cheam/fintech_pro2/OLS_H/OLS_H_y_pred_fullset.parquet')
pred_enet = pd.read_parquet('/home/cheam/fintech_pro2/enat/enat_recur_fullset.parquet')

# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

def diebold_mariano_test(y_true, y_pred1, y_pred2, h=1):
    # 计算预测误差
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2

    d = e1**2 - e2**2

    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)  
    # if d_var == 0:
    #     raise ValueError("Variance of the differences is zero. Cannot perform DM test.")

    # 计算 DM 统计量
    print(d_mean)
    print(d_var)
    dm_stat = d_mean / np.sqrt(d_var / len(d))

    # 计算 p 值
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return dm_stat, p_value

dm_stat, p_value = diebold_mariano_test(true['RET'], pred_ols['pred'], pred_ols3['pred'])
print(f"ols_ols3_DM Statistic: {dm_stat}")
dm_stat, p_value = diebold_mariano_test(true['RET'], pred_ols['pred'], pred_enet['pred'])
print(f"ols_enet_DM Statistic: {dm_stat}")
dm_stat, p_value = diebold_mariano_test(true['RET'], pred_ols3['pred'], pred_enet['pred'])
print(f"ols3_enet_DM Statistic: {dm_stat}")

# %% [markdown]
# # parameter

# %%
import numpy as np
from sklearn.linear_model import ElasticNet
import joblib  

para = pd.DataFrame(columns=['number'])
for i in range(30):
    year = 1987+i 
    name = '/home/cheam/fintech_pro2/enat/enet_recur_fullset_1957_'+ str(year) +'.pkl'
    #print(name)
    model = joblib.load(name)  
    coefficients = model.coef_
    non_zero_count = np.sum(coefficients != 0)
    para.loc[year] = non_zero_count

para


# %%
import seaborn as sns 
sns.set()
plt.figure(figsize=(10, 5))
plt.plot(para.index, para['number'], marker='o', linestyle='-', color='royalblue', linewidth=2, markersize=6)
#plt.gca().set_facecolor('lavender')  
plt.xlabel("Year")
plt.ylabel("Model complexity")
# plt.title("Tree Depth vs Year for XGBoost Model Complexity")
plt.grid(True)
plt.show()

# %% [markdown]
# # importance

# %%
bot = pd.read_parquet('/home/cheam/fintech_pro2/datashare/GKX_bottom20.parquet')
bot

# %%
import joblib
import pandas as pd

model = joblib.load('/home/cheam/fintech_pro2/OLS_H/OLS_H_full.pkl')  # 替换为你的文件名

features = bot.columns.drop(['DATE', 'permno', 'RET'])  # 替换为你的特征名称

importance_df = pd.DataFrame(index = features)
feature_importance = np.abs(model.coef_)

importance_sum = feature_importance.sum()
normalized_importance = feature_importance / importance_sum
#print(normalized_importance)
importance_df = pd.DataFrame(normalized_importance, index=features, columns=['Importance'])
importance_df = importance_df.sort_values('Importance', ascending=False)
importance_df.index.name = 'Feature'
importance_df.reset_index(inplace=True)
importance_df

# %%
for i in range(30):
    year = 1987+i 
    name = '/home/cheam/fintech_pro2/enat/enet_recur_fullset_1957_'+ str(year) +'.pkl'
    #print(name)
    model = joblib.load(name)  
    feature_importance = np.abs(model.coef_)
    importance_sum = feature_importance.sum()
    normalized_importance = feature_importance / importance_sum
    #print(normalized_importance)
    importance_df[year] = normalized_importance
    # print(importance_df)

importance_df.fillna(0, inplace=True)
importance_df = importance_df.sum(axis=1)/importance_df.sum(axis=1).sum()
importance_df = importance_df.sort_values(ascending=False)
importance_df = pd.DataFrame(importance_df,columns=['Importance'])
importance_df.index.name = 'Feature'
importance_df.reset_index(inplace=True)
importance_df

# %%
importance_df.to_parquet('/home/cheam/fintech_pro2/feature_importance/ols_full.parquet')

# %%
top_merged_importances_df = pd.read_parquet('/home/cheam/fintech_pro2/feature_importance/enet_bottom.parquet').sort_values('Importance', ascending=False)
top_merged_importances_df

# %%
origin_features = bot.columns.drop(['DATE', 'permno', 'RET']).tolist()[:94]
origin_features.append('sic')
sum_importance = pd.DataFrame(np.zeros(95), index = origin_features,columns = ['Importance'])
sum_importance.index.name = 'Feature'
for cols in origin_features:
    #print(cols)
    sum_importance.loc[cols] = top_merged_importances_df[top_merged_importances_df['Feature'].str.contains(cols)]['Importance'].sum()

sum_importance= sum_importance.sort_values('Importance', ascending=False)
sum_importance.reset_index(inplace=True)
sum_importance['Importance'] = sum_importance['Importance']/sum_importance['Importance'].sum()
sum_importance

# %%
import seaborn as sns
import pandas as pd
sns.set()

top_merged_importances_df = pd.read_parquet('/home/cheam/fintech_pro2/feature_importance/ols_full_sum_importance.parquet').sort_values('Importance', ascending=False)
# Select the top 20 feature importances
top_features_df = top_merged_importances_df.head(20)

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=top_features_df, palette='viridis')
#plt.title('Top 20 Feature Importances for ENet of bottomset')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# %%
df = pd.DataFrame(np.random.rand(3, 11), columns=['OLS3','ENet', 'PCA', 'PLS', 'RF', 'XGB', 'NN1','NN2', 'NN3', 'NN4', 'NN5'], index = ['All', 'Top', 'Bottom'])
df.iloc[0] = []
df.iloc[1] = []
df.iloc[2] = []
df

# %%
bar_width = 0.25
x = np.arange(len(df.columns))  
plt.bar(x - bar_width, df.iloc[0], width=bar_width, label='All', color='lightblue')
plt.bar(x, df.iloc[1], width=bar_width, label='Top', color='pink')
plt.bar(x + bar_width, df.iloc[2], width=bar_width, label='Bottom', color='lavender')
plt.xlabel('Model')
plt.ylabel('R2')
#plt.title('Combined Bar Chart for 3 Rows')
plt.xticks(x, df.columns)  
plt.legend()
plt.show()



