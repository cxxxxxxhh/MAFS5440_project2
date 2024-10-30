import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns 
sns.set()


n_components_list_top = []
n_components_list_bottom = []



#import data
PLS_y_top=data_top.loc[(data_top.index.get_level_values(0).year >= train_start_date+30) & (data_top.index.get_level_values(0).year < test_end_date)]
PLS_y_top=PLS_y_top.loc[:, ['RET']]
PLS_top_pred=pd.read_parquet('PLS_models/top/PLS_top_pred.parquet')
PLS_top_IMP=pd.read_parquet('PLS_models/top/PLS_top_g_importance.parquet')

PLS_y_bottom=data_bottom.loc[(data_bottom.index.get_level_values(0).year >= train_start_date+30) & (data_bottom.index.get_level_values(0).year < test_end_date)]
PLS_y_bottom=PLS_y_bottom.loc[:,['RET']]
PLS_bottom_pred=pd.read_parquet('PLS_models/bottom/PLS_bottom_pred.parquet')
PLS_bottom_IMP=pd.read_parquet('PLS_models/bottom/PLS_bottom_g_importance.parquet')


PLS_y_full=data_bottom.loc[(data_bottom.index.get_level_values(0).year >= train_start_date+30) & (data_bottom.index.get_level_values(0).year < test_end_date)]
PLS_y_full=PLS_y_full.loc[:,['RET']]
PLS_full_pred=pd.read_parquet('PLS_models/full/PLS_full_pred.parquet')
PLS_full_IMP=pd.read_parquet('PLS_models/full/PLS_full_importance.parquet')


#model complexity

n_components_list_top = []
n_components_list_bottom = []

for year in range(1987, 2017):
    model_path_top = f'PLS_models/top/PLS1957_{year}.pkl'
    model_path_bottom= f'PLS_models/bottom/PLS1957_{year}.pkl'
    pls_model_top = joblib.load(model_path_top)
    pls_model_bottom = joblib.load(model_path_bottom)
    n_components_list_top.append(pls_model_top.n_components)
    n_components_list_bottom.append(pls_model_bottom.n_components)
num_df = pd.DataFrame(n_components_list_top, index=range(1987, 2017), columns=['n_components'])
index=num_df.index
plt.figure(figsize=(10, 5))
plt.plot(index, n_components_list_top, label='n_components_top', marker='o')
plt.plot(index, n_components_list_bottom, label='n_components_bottom', marker='*')

plt.xlabel('Year')
plt.ylabel('Model Complexity')
#plt.title('yearly n_components')
plt.legend()
plt.grid(True)
plt.show()






#yearly_r2

calculator = R2Calculator()
PLS_top_yearly_r2 = calculator.cal_yearly_R2(PLS_top_pred, PLS_y_top)
PLS_bottom_yearly_r2 = calculator.cal_yearly_R2(PLS_bottom_pred, PLS_y_bottom)
PLS_full_yearly_r2  = calculator.cal_yearly_R2(PLS_full_pred, PLS_y_full)

index = PLS_top_yearly_r2.index

print('PLS_full_yearly_r2_',PLS_full_yearly_r2.index[-1],':',PLS_full_yearly_r2.iloc[-1])
print('PLS_top_yearly_r2_',PLS_top_yearly_r2.index[-1],':',PLS_top_yearly_r2.iloc[-1])
print('PLS_bottom_yearly_r2_',PLS_bottom_yearly_r2.index[-1],':',PLS_bottom_yearly_r2.iloc[-1])

plt.figure(figsize=(10, 5))
plt.plot(index, PLS_top_yearly_r2, label='PLS_top_yearly_r2', marker='o')
plt.plot(index, PLS_bottom_yearly_r2, label='PLS_bottom_yearly_r2', marker='x')
plt.plot(index, PLS_full_yearly_r2, label='PLS_full_yearly_r2', marker='x')


plt.xlabel('Date')
plt.ylabel('Values')
#plt.title('PLS Yearly R^2')
plt.legend()
plt.grid(True)
plt.show()


#monthly r2
PLS_full_monthly_r2 = calculator.cal_monthly_R2(PLS_full_pred, PLS_y_full)
PLS_top_monthly_r2 = calculator.cal_monthly_R2(PLS_top_pred, PLS_y_top)
PLS_bottom_monthly_r2 = calculator.cal_monthly_R2(PLS_bottom_pred, PLS_y_bottom)

print('PLS_full_monthly_r2_',PLS_full_monthly_r2.index[-1],':',PLS_full_monthly_r2.iloc[-1])
print('PLS_top_monthly_r2_',PLS_top_monthly_r2.index[-1],':',PLS_top_monthly_r2.iloc[-1])
print('PLS_bottom_monthly_r2_',PLS_bottom_monthly_r2.index[-1],':',PLS_bottom_monthly_r2.iloc[-1])



index = PLS_top_monthly_r2.index

plt.figure(figsize=(10, 5))


plt.plot(index, PLS_full_monthly_r2, label='PLS_full_monthly_r2')
plt.plot(index, PLS_top_monthly_r2, label='PLS_top_monthly_r2', marker='o')
plt.plot(index, PLS_bottom_monthly_r2, label='PLS_bottom_monthly_r2', marker='x')

plt.xlabel('month')
plt.ylabel('R^2')
plt.title('Monthly R^2')
plt.legend()
plt.grid(True)
plt.show()

#feature importance

from ML_Method import ML_method
MLM = ML_method(PLS_full_IMP)
PLS_IMP_G = MLM.calculate_importance()
PLS_IMP_G.set_index(PLS_IMP_G.columns[0], inplace=True)

merged_df=PLS_IMP_G
merged_df.index = merged_df.index.str.split('_').str[0]

df_grouped = merged_df.groupby(merged_df.index).sum()
df_grouped=df_grouped.sort_values(by='Importance', ascending=False)
top_features_df = df_grouped.head(20)

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=top_features_df, palette='viridis')
#plt.title('Top 20 Feature Importances for PLS')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()



