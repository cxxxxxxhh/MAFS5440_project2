import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# import data
data = pd.read_parquet('GKX_top20.parquet').set_index(['DATE','permno'])
year = data.index.get_level_values(0).year.unique().tolist()
#initialized
y_pred =[]
x_importance=[]



def R_oos(actual, predicted):
    actual, predicted =np.array(actual).flatten(),np.array(predicted).flatten()
    return 1 -(np.dot((actual-predicted),(actual-predicted)))/(np.dot(actual,actual))

class PCARegressor:
    def __init__(self):
        self.pca_best = None
        self.model_best = None
        self.best_components = 1

    def fit(self, X_train, y_train, X_valid, y_valid):
        self.pca_best = PCA(n_components=self.best_components).fit(X_train)
        X_train_pca = self.pca_best.transform(X_train)
        self.model_best = LinearRegression().fit(X_train_pca, y_train)

        best_score = mean_squared_error(y_valid, self.model_best.predict(self.pca_best.transform(X_valid)))

        for i in range(1, min(X_train.shape[1], 100)):  
            pca_temp = PCA(n_components=i).fit(X_train)
            X_train_pca_temp = pca_temp.transform(X_train)
            model_temp = LinearRegression().fit(X_train_pca_temp, y_train)

            score = mean_squared_error(y_valid, model_temp.predict(pca_temp.transform(X_valid)))

            if score < best_score:  
                self.best_components = i
                self.pca_best = pca_temp
                self.model_best = model_temp
                best_score = score

        return self

    def predict(self, X_new):
        if self.pca_best is None or self.model_best is None:
            raise Exception("The model has not been fitted yet. Please call fit() first.")

        X_new_pca = self.pca_best.transform(X_new)
        y_pred = self.model_best.predict(X_new_pca)
        return y_pred

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

        # 计算每个特征的重要性
        importances = np.abs(self.pca.components_).sum(axis=0)
        return importances

for i in range(30):

    train_start_date = year[0]
    valid_start_date = year[i]+18
    test_start_date = year[i]+18+12
    test_end_date= year[i]+18+12+1 
    #time period
    train_data = data.loc[(data.index.get_level_values(0).year >= train_start_date) & (data.index.get_level_values(0).year < valid_start_date)]
    valid_data =data.loc[(data.index.get_level_values(0).year >= valid_start_date) & (data.index.get_level_values(0).year < test_start_date)]
    test_data =data.loc[(data.index.get_level_values(0).year >= test_start_date) & (data.index.get_level_values(0).year < test_end_date)]



    #find the best parameters using valid_data
    pca_best = PCARegressor()
    pca_best.fit(train_data.drop(columns=['RET']),
                        train_data['RET'],
                        valid_data.drop(columns=['RET']),
                        valid_data['RET'])
    best_components = pca_best.best_components
    


    #using new model to train it
    combined_data = data.loc[(data.index.get_level_values(0).year >= train_start_date) & (data.index.get_level_values(0).year < test_start_date)]

    pca_best = SimplePCARegressor(n_components=best_components)

    pca_best.fit(combined_data.drop(columns=['RET']), combined_data['RET'])




    #save it
    name ='PCA_models/top/PCA'+ str(train_start_date) + '_'+ str(test_start_date) + '.pkl'
    joblib.dump(pca_best,name)    

    #prediction
    pred =pca_best.predict(test_data.drop(columns=['RET']))
    pred=pd.DataFrame(pred, index=test_data.index)
    pred.columns=['pred']
    y_pred.append(pred)


    #feature importance
    feature_importance = np.abs(pca_best.feature_importances()).flatten()
    feature_names = combined_data.drop(columns=['RET']).columns
    feature_importance / feature_importance.sum()
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df.set_index('Feature', inplace=True)
    importance_df['DATE'] = test_data.index[0][0]
    importance_df.set_index('DATE', append=True, inplace=True)
    importance_df = importance_df.reorder_levels(['DATE', 'Feature'])
    x_importance.append(importance_df)

    print(i,end='|')



y_pred_df=pd.concat(y_pred)
x_importance_df=pd.concat(x_importance)
print(R_oos(data['RET'][y_pred_df.index], y_pred_df['pred']))

y_pred_df.to_parquet('PCA_models/full/PCA_full_y_pred.parquet')
x_importance_df.to_parquet('PCA_models/full/PCA_full_x_importance.parquet')