# Import packages
import numpy as np
import pandas as pd
import math
import pickle
from datetime import datetime

from dateutil.relativedelta import relativedelta
import sklearn
import os
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid

import warnings
warnings.simplefilter(action='ignore', category=Warning)
import seaborn as sns
sns.set()
pd.options.mode.chained_assignment = None  # default='warn'
print("import packages successfully")

# List of feature names in stock_level_list
stock_level_list = [
    'mvel1', 'beta', 'betasq', 'chmom', 'dolvol',
    'idiovol', 'indmom', 'mom1m', 'mom6m', 'mom12m', 'mom36m', 'pricedelay',
    'turn', 'absacc', 'acc', 'age', 'agr', 'cashdebt', 'cashpr', 'cfp',
    'cfp_ia', 'chatoia', 'chcsho', 'chempia', 'chinv', 'chpmia', 'convind',
    'currat', 'depr', 'divi', 'divo', 'dy', 'egr', 'ep', 'gma', 'grcapx',
    'grltnoa', 'herf', 'hire', 'invest', 'lev', 'lgr', 'mve_ia', 'operprof',
    'orgcap', 'pchcapx_ia', 'pchcurrat', 'pchdepr', 'pchgm_pchsale',
    'pchquick', 'pchsale_pchinvt', 'pchsale_pchrect', 'pchsale_pchxsga',
    'pchsaleinv', 'pctacc', 'ps', 'quick', 'rd', 'rd_mve', 'rd_sale',
    'realestate', 'roic', 'salecash', 'saleinv', 'salerec', 'secured',
    'securedind', 'sgr', 'sin', 'sp', 'tang', 'tb', 'aeavol', 'cash',
    'chtx', 'cinvest', 'ear', 'nincr', 'roaq', 'roavol', 'roeq', 'rsup',
    'stdacc', 'stdcf', 'ms', 'baspread', 'ill', 'maxret', 'retvol',
    'std_dolvol', 'std_turn', 'zerotrade', 'bm', 'bm_ia'
]

def recursive_cv_split(data, date_column='DATE', train_start_year=1957, train_end_year=1974, 
                       val_start_year=1975, val_end_year=1986, test_start_year=1987, test_end_year=2016):
    train_indices_list = []
    val_indices_list = []
    test_indices_list = []

    # Initial train and validation periods
    train_start = datetime(train_start_year, 1, 31)
    train_end = datetime(train_end_year, 12, 31)
    val_start = datetime(val_start_year, 1, 31)
    val_end = datetime(val_end_year, 12, 31)
    test_start = datetime(test_start_year, 1, 31)
    test_end = datetime(test_start_year, 12, 31)

    # Ensure the test_end is within the data frame's date range
    max_date = data[date_column].max()
    while test_end.year <= max_date.year:
        # Train indices
        cur_train_indices = list(data[(data[date_column].dt.year >= train_start.year) &
                                      (data[date_column].dt.year <= train_end.year)].index)

        # Validation indices
        cur_val_indices = list(data[(data[date_column].dt.year >= val_start.year) &
                                    (data[date_column].dt.year <= val_end.year)].index)

        # Test indices
        cur_test_indices = list(data[(data[date_column].dt.year == test_start.year)].index)

        train_indices_list.append(cur_train_indices)
        val_indices_list.append(cur_val_indices)
        test_indices_list.append(cur_test_indices)

        # Update dates
        train_end += relativedelta(years=1)
        val_start += relativedelta(years=1)
        val_end += relativedelta(years=1)
        test_start += relativedelta(years=1)
        test_end += relativedelta(years=1)

    index_output = [(train, val, test) for train, val, test in zip(train_indices_list, val_indices_list, test_indices_list)]
    
    return index_output

df_all = pd.read_parquet("project2/GKX_fullset.parquet")
print("load data successfully")

#Copy date variable and stock id to use them as multindex
df_all["permno2"] = df_all["permno"].copy()
df_all["DATE2"] = df_all["DATE"].copy()
df_all = df_all.set_index(['DATE2','permno2'])

# Define the independent variables (X) and the dependent variable (y)
features = df_all.columns[~df_all.columns.isin(["RET", "permno"])].tolist()
X = df_all[features]
y = df_all["RET"]

# Empty containers to save results from each window
predictions = []
y_test_list = []
dates = []
dic_r2_all = {}

dic_max_depth_all = {}
y_pred = []

param_grid = {'max_depth': [2, 3, 4], 
              'n_estimators': [50, 200, 300], 
              "learning_rate": [0.01, 0.1]}
# Convert grid to list containing all possible hyperparameter combinations to iterate over
grid = list(ParameterGrid(param_grid))
# Empty container to save the objective loss function (MSE) for each hyperparameter combination
mse = np.full((len(grid), 1), np.nan, dtype=np.float32)

# Initialize empty list to store feature importances
all_feature_importances = []

# Define the path where models will be saved
model_dir = "XGB_models"
os.makedirs(model_dir, exist_ok=True)

# Define the path where dictionaries will be saved
dictionary_dir = "XGB_dictionaries"
os.makedirs(dictionary_dir, exist_ok=True)

# Iterate over each window generated by the recursive CV splitter
for i, (train_index, val_index, test_index) in enumerate(recursive_cv_split(X, date_column="DATE")):
    # Split data into train, validation, and test sets
    X_train = X.loc[train_index].drop('DATE', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('DATE', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('DATE', axis=1)
    y_test = y.loc[test_index]
    print(y_test)
    
    # Loop over the list containing all hyperparameter combinations
    for j in range(len(grid)):
        XGB_val = XGBRegressor(objective='reg:squarederror', 
                               max_depth=grid[j]["max_depth"],
                               learning_rate=grid[j]["learning_rate"],
                               n_estimators=grid[j]["n_estimators"])
        
        XGB_val.fit(X_train, y_train)
        Yval_predict = XGB_val.predict(X_val)
        
        mse[j, 0] = ((y_val - Yval_predict) ** 2).mean()
        print("MSE for combination", j, ":", mse[j, 0])
    
    # The optimal combination of hyperparameters is the one that causes the lowest loss
    optim_param = grid[np.argmin(mse)]
    print("optim_param for ith recursive: ", optim_param)
    
    # Fit again using the train and validation set and the optimal value for each hyperparameter
    XGB = XGBRegressor(objective='reg:squarederror',
                       max_depth=optim_param["max_depth"],
                       learning_rate=optim_param["learning_rate"],
                       n_estimators=optim_param["n_estimators"])
    
    XGB.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    
    # Save the model using pickle
    model_path = os.path.join(model_dir, f"model{i}.pkl")
    with open(model_path, 'wb') as model_file:
        pickle.dump(XGB, model_file)
    print(f"Model saved to {model_path}")
    
    # Predict on test set
    preds = XGB.predict(X_test)

    # Create dataframe for predictions and append it to the list
    pred = pd.DataFrame(preds, index=y_test.index, columns=['pred'])
    y_pred.append(pred)
    
    # Save predictions, dates, and the true values of the dependent variable to list
    predictions.append(preds)
    dates.append(y_test.index)
    y_test_list.append(y_test)
    
    # Calculate OOS model performance for the current window
    r2 = 1 - ((y_test - preds) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
    # Save OOS model performance and the respective month to dictionary
    dic_r2_all["r2." + str(y_test.index)] = r2
    
    # Save the number of predictors randomly considered as potential split variables
    dic_max_depth_all["feat." + str(y_test.index)] = optim_param["max_depth"]
    
    # Record feature importances
    feature_importances = pd.Series(XGB.feature_importances_, index=X_train.columns)
    all_feature_importances.append(feature_importances)
    
# Concatenate to get results over the whole OOS test period
predictions_all = np.concatenate(predictions, axis=0)
y_test_list_all = np.concatenate(y_test_list, axis=0)
dates_all = np.concatenate(dates, axis=0)

y_pred_all = pd.concat(y_pred)
y_pred_all.to_parquet('XGB_y_pred.parquet')
print(f"Predictions saved successfully!")

# Calculate OOS model performance over the entire test period
R2OOS_XGB = 1 - ((y_test_list_all - predictions_all) ** 2).sum() / ((y_test_list_all - y_test_list_all.mean()) ** 2).sum()
print("R2OOS XGBoost: ", R2OOS_XGB)

# save R2OOS_RF to text file
with open('XGBoost_r2oos.txt', 'w') as file:
    file.write(f"Overall OOS R-squared for XGBoost: {R2OOS_XGB}\n")

# Save dictionaries to file
with open(os.path.join(dictionary_dir, 'dic_r2_all.pkl'), 'wb') as handle:
    pickle.dump(dic_r2_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("dic_r2_all saved.")

with open(os.path.join(dictionary_dir, 'dic_max_depth_all.pkl'), 'wb') as handle:
    pickle.dump(dic_max_depth_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("dic_max_depth_all saved.")

# Average feature importances
avg_feature_importances = pd.concat(all_feature_importances, axis=1).mean(axis=1)

# Create DataFrame
feature_importances_df = pd.DataFrame({
    'Feature': avg_feature_importances.index,
    'Importance': avg_feature_importances.values
}).sort_values(by='Importance', ascending=False)

# Create directory if it does not exist
feature_importance_dir = 'XGB_feature_importance'
os.makedirs(feature_importance_dir, exist_ok=True)

# Save the feature importances DataFrame to CSV
feature_importances_path = os.path.join(feature_importance_dir, 'feature_importances_all.csv')
feature_importances_df.to_csv(feature_importances_path, index=False)

# Merge the importance of features in stock_level_list
merged_importances_stock_level = {}
for stock_level_feature in stock_level_list:
    pattern = f'{stock_level_feature}_'
    matched_features = feature_importances_df[
        feature_importances_df['Feature'].str.contains(pattern, case=False)
    ]
    if not matched_features.empty:
        merged_importances_stock_level[stock_level_feature] = matched_features['Importance'].sum()

# Merge the importance of features containing "sic2"
matched_features_sic2 = feature_importances_df[
    feature_importances_df['Feature'].str.contains('sic2', case=False)
]
if not matched_features_sic2.empty:
    merged_importances_stock_level['sic2'] = matched_features_sic2['Importance'].sum()

# Create a new DataFrame containing merged feature importances
merged_importances_df = pd.DataFrame.from_dict(
    merged_importances_stock_level, orient='index', columns=['Importance']
).reset_index().rename(columns={'index': 'Feature'})

# Sort the DataFrame
merged_importances_df = merged_importances_df.sort_values(by='Importance', ascending=False)

# Print the merged feature importances
print(merged_importances_df.head(20))

features_df = merged_importances_df.head(20)

# plot
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=features_df, palette='viridis')
plt.title('Top 20 Feature Importances (XGBoost)')
plt.xlabel('Importance')
plt.ylabel('Feature')

output_path = "feature_importances_top_20(XGB).png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')

plt.show()

