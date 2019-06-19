#imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras import backend
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings 
import tensorflow as tf
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBRegressor
import os
from sklearn.externals.joblib import parallel_backend

print(os.listdir("/lhome/nriahid/Documents/automl2019-kaggle/data"))


def get_data():
    #get train data
    train_data_path ='/lhome/nriahid/Documents/automl2019-kaggle/data/traindata.csv'
    train_label_path = '/lhome/nriahid/Documents/automl2019-kaggle/data/traindata_label.csv'
    train_x = pd.read_csv(train_data_path)
    train_y = pd.read_csv(train_label_path)
    
    #get test data
    test_data_path ='/lhome/nriahid/Documents/automl2019-kaggle/data/testdata.csv'
    test_x = pd.read_csv(test_data_path)
    
    return train_x , train_y, test_x

def get_combined_data():
  #reading train data
  train_x , train_y,  test_x = get_data()

  combined = train_x.append(test_x)
  combined.reset_index(inplace=True)
  combined.drop(['index'], inplace=True, axis=1)
  return combined, train_y

#Load train and test data into pandas DataFrames
train_x, target ,test_x = get_data()

#Combine train and test data to process them together
combined, target = get_combined_data()

def get_cols_with_no_nans(df,col_type):
    '''
    Arguments :
    df : The dataframe to process
    col_type : 
          num : to only get numerical columns with no nans
          no_num : to only get nun-numerical columns with no nans
          all : to get any columns with no nans    
    '''
    if (col_type == 'num'):
        predictors = df.select_dtypes(exclude=['object'])
    elif (col_type == 'no_num'):
        predictors = df.select_dtypes(include=['object'])
    elif (col_type == 'all'):
        predictors = df
    else :
        print('Error : choose a type (num, no_num, all)')
        return 0
    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans

num_cols = get_cols_with_no_nans(combined , 'num')
cat_cols = get_cols_with_no_nans(combined , 'no_num')

print ('Number of numerical columns with no nan values :',len(num_cols))
print ('Number of nun-numerical columns with no nan values :',len(cat_cols))

def oneHotEncode(df,colNames):
    for col in colNames:
        if( df[col].dtype == np.dtype('object')):
            dummies = pd.get_dummies(df[col],prefix=col)
            df = pd.concat([df,dummies],axis=1)

            #drop the encoded column
            df.drop([col],axis = 1 , inplace=True)
    return df
    

print('There were {} columns before encoding categorical features'.format(combined.shape[1]))
combined = oneHotEncode(combined, cat_cols)
print('There are {} columns after encoding categorical features'.format(combined.shape[1]))


def split_combined():
    global combined
    train = combined[:16325]
    test = combined[16325:]

    return train , test 

train, test = split_combined()

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.05, random_state = 14)


rf = RandomForestRegressor()

parameters = {
 'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

n_iter_search = 5
rf_grid = RandomizedSearchCV( rf,
                        parameters,
                        n_iter=n_iter_search,
                        cv = 2,
                        n_jobs = -1,
                        verbose=True)

with parallel_backend('threading'):
    rf_grid.fit(train_X,train_y)


print(rf_grid.best_score_)
print(rf_grid.best_params_)


# Get the mean absolute error on the validation data
predicted_prices = rf_grid.predict(val_X)
MAE = mean_absolute_error(val_y , predicted_prices)
print('Random forest validation MAE = ', MAE)


def make_submission(prediction, sub_name):
  my_submission = pd.DataFrame({'ID':pd.read_csv('/lhome/nriahid/Documents/automl2019-kaggle/data/testdata.csv').index,'AveragePrice':prediction})
  my_submission.to_csv('/lhome/nriahid/Documents/automl2019-kaggle/result/{}'.format(sub_name),index=False)
  print('A submission file has been made')

predicted_prices = rf_grid.predict(test)
make_submission(predicted_prices,'Submission(RF).csv')
