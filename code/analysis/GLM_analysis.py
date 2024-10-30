#import libraries
import warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import joblib
import seaborn as sns 
sns.set()
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from ML_Method import R2Calculator


#import original data
data_top = pd.read_parquet('DATA/GKX_top20.parquet').set_index(['DATE','permno'])
data_bottom=pd.read_parquet('DATA/GKX_bottom20.parquet').set_index(['DATE','permno'])
data_full=pd.read_parquet('DATA/GKX_1987.parquet').set_index(['DATE','permno'])
year = data_top.index.get_level_values(0).year.unique().tolist()
i=29
train_start_date = year[0]
valid_start_date = year[i] + 18
test_start_date = valid_start_date + 12
test_end_date = test_start_date + 1



class GroupLassoHuber(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, epsilon=1.35, max_iter=300, tol=1e-05):
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.huber = HuberRegressor(epsilon=self.epsilon, max_iter=self.max_iter, tol=self.tol)

    def fit(self, X, y):
        self.huber.fit(X, y)
        n_features = X.shape[1] // 3  # Number of original features
        coefs = self.huber.coef_.reshape(n_features, 3)
        norms = np.linalg.norm(coefs, axis=1)
        penalties = self.alpha * norms
        self.coef_ = coefs / (1 + penalties[:, np.newaxis])
        self.intercept_ = self.huber.intercept_
        return self

    def predict(self, X):
        return self.intercept_ + X @ self.coef_.flatten()

class SplineLinearModel:
    def __init__(self, alpha=1.0):
        self.model = GroupLassoHuber(alpha=alpha)

    def transform(self, X):
        # Assuming X is a DataFrame or 2D array with shape (n_samples, n_features)
        X_transformed = np.hstack([
            X,  # Linear term
            X ** 2,  # Spline term z^2
            X ** 3   # Spline term z^3
        ])
        return X_transformed

    def fit(self, X, y):
        X_transformed = self.transform(X)
        self.model.fit(X_transformed, y)

    def predict(self, X):
        X_transformed = self.transform(X)
        return self.model.predict(X_transformed)

    def feature_importance(self, feature_names):
        # Get the coefficients from the model
        coefs = self.model.coef_
        n_features = len(feature_names)
        
        # Calculate the importance for each original feature as the sum of its linear, quadratic, and cubic terms
        importance = {
            feature_names[i]: coefs[i, 0] + coefs[i, 1] + coefs[i, 2]
            for i in range(n_features)
        }
        
        # Transform the importance dictionary into a DataFrame
        importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
        
        return importance_df

    def R_oos(self, actual, predicted):
        actual, predicted = np.array(actual).flatten(), np.array(predicted).flatten()
        return 1 - (np.dot((actual - predicted), (actual - predicted))) / (np.dot(actual, actual))

def count_non_zero_features(spline_model, threshold=1e-4):
    group_lasso_model = spline_model.model
    coefs = group_lasso_model.coef_
    non_zero_features = np.sum(np.all(np.abs(coefs) > threshold, axis=1))
    return non_zero_features


#model complexity
for year in range(1987, 2017):
    # 构建文件名
    model_path_top = f'GLM_models/top/GLM1957_{year}.pkl'
    # 加载GLM模型
    GLM_model_top = joblib.load(model_path_top)
    non_zero_features = count_non_zero_features(GLM_model_top)
    print(f"the number of nonzero characteristics: {non_zero_features}。")


n_components_list_top = []
n_components_list_bottom = []

for year in range(1987, 2017):
    model_path_top = f'GLM_models/top/GLM1957_{year}.pkl'
    model_path_bottom= f'GLM_models/bottom/GLM1957_{year}.pkl'
    GLM_model_top = joblib.load(model_path_top)
    GLM_model_bottom = joblib.load(model_path_bottom)
    GLM_top_N=count_non_zero_features(GLM_model_top)
    GLM_bottom_N=count_non_zero_features(GLM_model_bottom)

    n_components_list_top.append(GLM_top_N)
    n_components_list_bottom.append(GLM_bottom_N)

num_df = pd.DataFrame(n_components_list_top, index=range(1987, 2017), columns=['n_components'])
index=num_df.index
plt.figure(figsize=(10, 5))
plt.plot(index, n_components_list_top, label='n_components_top', marker='o')
plt.plot(index, n_components_list_bottom, label='n_components_bottom', marker='*')

plt.xlabel('Year')
plt.ylabel('Model Complexity')
plt.legend()
plt.grid(True)
plt.show()


#yearly r^2
import seaborn as sns 
sns.set()
GLM_y_top=data_top.loc[(data_top.index.get_level_values(0).year >= train_start_date+30) & (data_top.index.get_level_values(0).year < test_end_date)]
GLM_y_top=GLM_y_top.loc[:, ['RET']]
GLM_top_pred=pd.read_parquet('GLM_models/top/GLM_top_pred.parquet')
GLM_top_IMP=pd.read_parquet('GLM_models/top/GLM_top_g_importance.parquet')

GLM_y_bottom=data_bottom.loc[(data_bottom.index.get_level_values(0).year >= train_start_date+30) & (data_bottom.index.get_level_values(0).year < test_end_date)]
GLM_y_bottom=GLM_y_bottom.loc[:,['RET']]
GLM_bottom_pred=pd.read_parquet('GLM_models/bottom/GLM_bottom_pred.parquet')
GLM_bottom_IMP=pd.read_parquet('GLM_models/bottom/GLM_bottom_g_importance.parquet')

calculator = R2Calculator()
GLM_top_yearly_r2 = calculator.cal_yearly_R2(GLM_top_pred, GLM_y_top)
GLM_bottom_yearly_r2 = calculator.cal_yearly_R2(GLM_bottom_pred, GLM_y_bottom)

index = GLM_top_yearly_r2.index

print('GLM_top_yearly_r2_',GLM_top_yearly_r2.index[-1],':',GLM_top_yearly_r2.iloc[-1])
print('GLM_bottom_yearly_r2_',GLM_bottom_yearly_r2.index[-1],':',GLM_bottom_yearly_r2.iloc[-1])

plt.figure(figsize=(10, 5))
plt.plot(index, GLM_top_yearly_r2, label='GLM_top_yearly_r2', marker='o')
plt.plot(index, GLM_bottom_yearly_r2, label='GLM_bottom_yearly_r2', marker='x')


plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()


#montly r2

GLM_top_monthly_r2 = calculator.cal_monthly_R2(GLM_top_pred, GLM_y_top)
GLM_bottom_monthly_r2 = calculator.cal_monthly_R2(GLM_bottom_pred, GLM_y_bottom)

print('GLM_top_monthly_r2_',GLM_top_monthly_r2.index[-1],':',GLM_top_monthly_r2.iloc[-1])
print('GLM_bottom_monthly_r2_',GLM_bottom_monthly_r2.index[-1],':',GLM_bottom_monthly_r2.iloc[-1])

index = GLM_top_monthly_r2.index

plt.figure(figsize=(10, 5))

plt.plot(index, GLM_top_monthly_r2, label='GLM_top_monthly_r2', marker='o')
plt.plot(index, GLM_bottom_monthly_r2, label='GLM_bottom_monthly_r2', marker='x')

plt.xlabel('month')
plt.ylabel('R^2')
plt.title('Monthly R^2')
plt.legend()
plt.grid(True)
plt.show()



# feature importance
from ML_Method import ML_method
MLM = ML_method(GLM_top_IMP)
GLM_top_IMP = MLM.calculate_importance()
GLM_top_IMP.set_index(GLM_top_IMP.columns[0], inplace=True)

merged_df=GLM_top_IMP
merged_df.index = merged_df.index.str.split('_').str[0]

df_grouped = merged_df.groupby(merged_df.index).sum()
df_grouped=df_grouped.sort_values(by='Importance', ascending=False)
top_features_df = df_grouped.head(20)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=top_features_df, palette='viridis')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

