
####  import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from ML_Method import R2Calculator


#import original data
data_top = pd.read_parquet('DATA/GKX_top20.parquet').set_index(['DATE','permno'])
data_bottom=pd.read_parquet('DATA/GKX_bottom20.parquet').set_index(['DATE','permno'])
data_full=pd.read_parquet('DATA/GKX_1987.parquet').set_index(['DATE','permno'])
year = data_top.index.get_level_values(0).year.unique().tolist()
#define time
i=29
train_start_date = year[0]
valid_start_date = year[i] + 18
test_start_date = valid_start_date + 12
test_end_date = test_start_date + 1


#import data
PCR_y_top=data_top.loc[(data_top.index.get_level_values(0).year >= train_start_date+30) & (data_top.index.get_level_values(0).year < test_end_date)]
PCR_y_top=PCR_y_top.loc[:, ['RET']]
PCR_top_pred=pd.read_parquet('PCA_models/top/PCA_top_pred.parquet')
PCR_top_IMP=pd.read_parquet('PCA_models/top/PCA_top_g_importance.parquet')

PCR_y_bottom=data_bottom.loc[(data_bottom.index.get_level_values(0).year >= train_start_date+30) & (data_bottom.index.get_level_values(0).year < test_end_date)]
PCR_y_bottom=PCR_y_bottom.loc[:,['RET']]
PCR_bottom_pred=pd.read_parquet('PCA_models/bottom/PCA_bottom_pred.parquet')
PCR_bottom_IMP=pd.read_parquet('PCA_models/bottom/PCA_bottom_g_importance.parquet')


PCR_y_full=data_bottom.loc[(data_bottom.index.get_level_values(0).year >= train_start_date+30) & (data_bottom.index.get_level_values(0).year < test_end_date)]
PCR_y_full=PCR_y_full.loc[:,['RET']]
PCR_full_pred=pd.read_parquet('PCA_models/full/PCA_full_y_pred.parquet')
PCR_full_IMP=pd.read_parquet('PCA_models/full/PCA_full_x_importance.parquet')


#yearly r^2
import seaborn as sns 
sns.set()
calculator = R2Calculator()
PCR_top_yearly_r2 = calculator.cal_yearly_R2(PCR_top_pred, PCR_y_top)
PCR_bottom_yearly_r2 = calculator.cal_yearly_R2(PCR_bottom_pred, PCR_y_bottom)
PCR_full_yearly_r2  = calculator.cal_yearly_R2(PCR_full_pred, PCR_y_full)

index = PCR_top_yearly_r2.index

print('PCR_full_yearly_r2_',PCR_full_yearly_r2.index[-1],':',PCR_full_yearly_r2.iloc[-1])
print('PCR_top_yearly_r2_',PCR_top_yearly_r2.index[-1],':',PCR_top_yearly_r2.iloc[-1])
print('PCR_bottom_yearly_r2_',PCR_bottom_yearly_r2.index[-1],':',PCR_bottom_yearly_r2.iloc[-1])

plt.figure(figsize=(10, 5))
plt.plot(index, PCR_top_yearly_r2, label='PCR_top_yearly_r2', marker='o')
plt.plot(index, PCR_bottom_yearly_r2, label='PCR_bottom_yearly_r2', marker='x')
plt.plot(index, PCR_full_yearly_r2, label='PCR_full_yearly_r2', marker='x')


plt.xlabel('Date')
plt.ylabel('Values')
#plt.title('PCR Yearly R^2')
plt.legend()
plt.grid(True)
plt.show()




#monthly r^2
PCR_full_monthly_r2 = calculator.cal_monthly_R2(PCR_full_pred, PCR_y_full)
PCR_top_monthly_r2 = calculator.cal_monthly_R2(PCR_top_pred, PCR_y_top)
PCR_bottom_monthly_r2 = calculator.cal_monthly_R2(PCR_bottom_pred, PCR_y_bottom)

print('PCR_full_monthly_r2_',PCR_full_monthly_r2.index[-1],':',PCR_full_monthly_r2.iloc[-1])
print('PCR_top_monthly_r2_',PCR_top_monthly_r2.index[-1],':',PCR_top_monthly_r2.iloc[-1])
print('PCR_bottom_monthly_r2_',PCR_bottom_monthly_r2.index[-1],':',PCR_bottom_monthly_r2.iloc[-1])



index = PCR_top_monthly_r2.index

plt.figure(figsize=(10, 5))


plt.plot(index, PCR_full_monthly_r2, label='PCR_full_monthly_r2')
plt.plot(index, PCR_top_monthly_r2, label='PCR_top_monthly_r2', marker='o')
plt.plot(index, PCR_bottom_monthly_r2, label='PCR_bottom_monthly_r2', marker='x')

plt.xlabel('month')
plt.ylabel('R^2')
plt.title('Monthly R^2')
plt.legend()
plt.grid(True)
plt.show()


#model complexity

class SimplePCARegressor:
    def __init__(self, n_components=1):
        self.n_components = n_components
        self.pca = None
        self.model = None

    def fit(self, X_train, y_train):
        self.pca = PCA(n_components=self.n_components).fit(X_train)
        X_train_pca = self.pca.transform(X_train)
        self.model = LinearRegression().fit(X_train_pca, y_train)
        return self

    def predict(self, X_new):
        if self.pca is None or self.model is None:
            raise Exception("The model has not been fitted yet. Please call fit() first.")

        X_new_pca = self.pca.transform(X_new)
        y_pred = self.model.predict(X_new_pca)
        return y_pred

    def feature_importances(self):
        if self.pca is None:
            raise Exception("The model has not been fitted yet. Please call fit() first.")

        importances = np.abs(self.pca.components_).sum(axis=0)
        return importances

n_components_list_top = []
n_components_list_bottom = []
n_components_list_full = []

for year in range(1987, 2017):
    model_path_top = f'PCA_models/top/PCA1957_{year}.pkl'
    model_path_bottom = f'PCA_models/bottom/PCA1957_{year}.pkl'
    model_path_full = f'PCA_models/full/PCA_full_1957_{year}.pkl'
    pca_model_top = joblib.load(model_path_top)
    pca_model_bottom = joblib.load(model_path_bottom)
    pca_model_full = joblib.load(model_path_full)
    n_components_list_top.append(pca_model_top.n_components)
    n_components_list_bottom.append(pca_model_bottom.n_components)
    n_components_list_full.append(pca_model_full.n_components)
num_df=pd.DataFrame(n_components_list_full)



year = data_top.index.get_level_values(0).year.unique().tolist()[30:]

index =year

plt.figure(figsize=(10, 5))
plt.plot(index, n_components_list_top, label='n_components_top', marker='o')
plt.plot(index, n_components_list_bottom, label='n_components_bottom', marker='x')
plt.plot(index, n_components_list_full, label='n_components_full', marker='*')

plt.xlabel('Year')
plt.ylabel('Model Complexity')
plt.legend()
plt.grid(True)
plt.show()

n_components_list_full[-1]


# importance feature

from ML_Method import ML_method
MLM = ML_method(PCR_full_IMP)
PCR_IMP_G = MLM.calculate_importance()
PCR_IMP_G.set_index(PCR_IMP_G.columns[0], inplace=True)
merged_df=PCR_IMP_G
merged_df.index = merged_df.index.str.split('_').str[0]
df_grouped = merged_df.groupby(merged_df.index).sum()
df_grouped=df_grouped.sort_values(by='Importance', ascending=False)
# Select the top 20 feature importances
top_features_df =df_grouped.head(20)

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=top_features_df, palette='viridis')
#plt.title('Top 20 Feature Importances for PCR')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

