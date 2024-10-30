import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression




# import data
data = pd.read_parquet('GKX_top20.parquet').set_index(['DATE','permno'])
year = data.index.get_level_values(0).year.unique().tolist()
#initialized
y_pred =[]
x_importance=[]




def PLSReg(X_train, y_train, X_valid, y_valid):
    best_pls_components = 1
    pls_best = PLSRegression(n_components=best_pls_components).fit(X_train, y_train)

    for i in range(1,min(X_train.shape[1],100)):  
        pls_temp = PLSRegression(n_components=i).fit(X_train, y_train)
        score = pls_temp.score(X_valid, y_valid)

        if score > pls_best.score(X_valid, y_valid):
            best_pls_components = i
            pls_best = pls_temp  

        #print(f"Components: {i}, Score: {score:.4f}")

    #print(f"\nBest number of components: {best_pls_components}")
    return pls_best


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
    pls_best=PLSReg(train_data.drop(columns=['RET']),
                      train_data['RET'],
                      valid_data.drop(columns=['RET']),
                      valid_data['RET'])
    best_params = pls_best.get_params()
    combined_data = data.loc[(data.index.get_level_values(0).year >= train_start_date) & (data.index.get_level_values(0).year < test_start_date)]
    pls_best = PLSRegression(**best_params)
    pls_best.fit(combined_data.drop(columns=['RET']), combined_data['RET'])

    name ='PLS_models/top/PLS'+ str(train_start_date) + '_'+ str(test_start_date) + '.pkl'
    joblib.dump(pls_best,name)    


    #prediction
    pred =pls_best.predict(test_data.drop(columns=['RET']))
    pred=pd.DataFrame(pred, index=test_data.index)
    pred.columns=['pred']
    y_pred.append(pred)


    #feature importance
    feature_importance = np.abs(pls_best.coef_).flatten()
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



y_pred_df.to_parquet('PLS_models/top/PLS_full_y_pred.parquet')
x_importance_df.to_parquet('PLS_models/top/PLS_full_x_importance.parquet')
