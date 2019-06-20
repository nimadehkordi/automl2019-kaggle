#imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import GridSearchCV
import seaborn as sb
import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBRegressor
import os
from sklearn.externals.joblib import parallel_backend


print(os.listdir("../data"))


def get_data():
    #get train data
    train_data_path ='../data/traindata.csv'
    train_label_path = '../data/traindata_label.csv'
    train_x = pd.read_csv(train_data_path)
    train_y = pd.read_csv(train_label_path)
    
    #get test data
    test_data_path ='../data/testdata.csv'
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

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.25, random_state = 14)

xgb1 = XGBRegressor()

parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, .07, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], #so called `eta` value
              'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
              'min_child_weight': [1, 3, 5, 7],
              'gamma' : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.3, 0.4, 0.5 , 0.7],
              'n_estimators': [500]}


xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = -1,
                        verbose=True)

with parallel_backend('threading'):
    xgb_grid.fit(train_X,train_y , verbose=False)


print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

# Get the mean absolute error on the validation data :
XGBpredictions = xgb_grid.predict(val_X)
MAE = mean_absolute_error(val_y , XGBpredictions)
print('XGBoost validation MAE = ',MAE)



def make_submission(prediction, sub_name):
  my_submission = pd.DataFrame({'ID':pd.read_csv('../data/testdata.csv').index,'AveragePrice':prediction})
  my_submission.to_csv('../result/{}'.format(sub_name),index=False)
  print('A submission file has been made')

XGBpredictions = xgb_grid.predict(test)
make_submission(XGBpredictions,'Submission(XGB).csv')