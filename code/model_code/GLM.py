import warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import joblib
warnings.filterwarnings("ignore", category=ConvergenceWarning)


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




# import data
data = pd.read_parquet('GKX_top20.parquet').set_index(['DATE','permno'])
year = data.index.get_level_values(0).year.unique().tolist()
#initialized
y_pred =[]
x_importance=[]


train_start_date = year[0]
for i in range(30):
    valid_start_date = year[i] + 18
    test_start_date = valid_start_date + 12
    test_end_date = test_start_date + 1
    train_data = data.loc[(data.index.get_level_values(0).year >= train_start_date) & (data.index.get_level_values(0).year < valid_start_date)]
    valid_data = data.loc[(data.index.get_level_values(0).year >= valid_start_date) & (data.index.get_level_values(0).year < test_start_date)]
    test_data = data.loc[(data.index.get_level_values(0).year >= test_start_date) & (data.index.get_level_values(0).year < test_end_date)]


    # Define the parameter grid for alpha (lambda)
    lambdas = np.logspace(-6, -4, num=5)

    # Initialize variables to store the best lambda and best R_oos value
    best_lambda = None
    best_r_oos_value = -np.inf

    # Iterate over each lambda value to find the best one
    for j, alpha in enumerate(lambdas):
        model = SplineLinearModel(alpha=alpha)
        
        # Train the model on train_data
        model.fit(train_data.drop(columns=['RET']), train_data['RET'])
        
        # Predict on valid_data
        predictions = model.predict(valid_data.drop(columns=['RET']))
        
        # Calculate R_oos value on valid_data
        r_oos_value = model.R_oos(valid_data['RET'], predictions)
        
        # Update best_lambda if current R_oos value is better
        if r_oos_value > best_r_oos_value:
            best_r_oos_value = r_oos_value
            best_lambda = alpha
        

    # retrain models
    glm_best = SplineLinearModel(alpha=best_lambda)
    glm_best.fit(combined_data.drop(columns=['RET']), combined_data['RET'])
    # Example usage:
    # Predict using the trained final model
    predictions_combined = glm_best.predict(test_data.drop(columns=['RET']))
    # Compare predictions with the real values using R_oos function
    real_values_combined = test_data['RET']
    r_oos_value_combined = glm_best.R_oos(real_values_combined, predictions_combined)
   
    
    combined_data = data.loc[(data.index.get_level_values(0).year >= train_start_date) & (data.index.get_level_values(0).year < test_start_date)]
    glm_best.fit(combined_data.drop(columns=['RET']), combined_data['RET'])

    # save the model
    name = f'GLM_models/top/GLM{train_start_date}_{test_start_date}.pkl'
    joblib.dump(glm_best, name)
    

    # predict
    pred = predictions_combined
    pred = pd.DataFrame(pred, index=test_data.index, columns=['pred'])
    y_pred.append(pred)
    # Show feature importance
    feature_names_combined = combined_data.drop(columns=['RET']).columns
    feature_importance_df_combined = glm_best.feature_importance(feature_names_combined)
    importance_df=feature_importance_df_combined
    importance_df.set_index('Feature', inplace=True)
    importance_df['DATE'] = test_data.index[0][0]
    importance_df.set_index('DATE', append=True, inplace=True)
    importance_df = importance_df.reorder_levels(['DATE', 'Feature'])
    
    x_importance.append(importance_df)
    
    print(i, end='|')



y_pred_df=pd.concat(y_pred)
x_importance_df=pd.concat(x_importance)
y_pred_df.to_parquet('GLM_models/top/GLM_top_y_pred.parquet')
x_importance_df.to_parquet('GLM_models/top/GLm_top_x_importance.parquet')