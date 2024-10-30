import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ML_method:
    def __init__(self, x_importance_df):
        self.x_importance_df = x_importance_df

    def calculate_importance(self):
        # Group by 'Feature' and calculate the mean of each feature across different 'DATE's
        igb = self.x_importance_df.groupby('Feature').mean()  # importance group by
        IGB = igb.sort_values(by='Importance', ascending=False)
        
        # Add a new column with the index values and place it to the left of the 'Value' column
        IGB.insert(0, 'Feature', IGB.index)
        IGB.reset_index(drop=True, inplace=True)
        
        # Normalize the 'Importance' column
        IGB['Importance'] = IGB['Importance'] / IGB['Importance'].abs().sum()        
        
        return IGB
    

class R2Calculator:
    @staticmethod
    def R_oos(actual, predicted):
        actual, predicted = np.array(actual).flatten(), np.array(predicted).flatten()
        return 1 - (np.dot((actual - predicted), (actual - predicted))) / (np.dot(actual, actual))

    def cal_monthly_R2(self, pred, true):
        merged = pd.merge(true, pred, left_index=True, right_index=True)
        monthly_r_squared = merged.groupby(level=0).apply(lambda group: self.R_oos(group['RET'], group['pred']))
        return monthly_r_squared

    def cal_yearly_R2(self, pred, true):
        merged = pd.merge(true, pred, left_index=True, right_index=True)
        yearly_r_squared = merged.groupby(merged.index.get_level_values(0).year).apply(lambda group: self.R_oos(group['RET'], group['pred']))
        return yearly_r_squared
    def plot_r_squared(self, r_squared_dict, time_index):
        plt.figure(figsize=(10, 6))
        for label, r_squared_values in r_squared_dict.items():
            plt.plot(time_index, r_squared_values, marker='o', linestyle='-', label=label)
        plt.xlabel('Time')
        plt.ylabel('R-squared')
        plt.title('R-squared Values Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()