# imports
import logging
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.keras import backend
except:
    raise ImportError("For this example you need to install keras.")

try:
    import torchvision
    import torchvision.transforms as transforms
except:
    raise ImportError("For this example you need to install pytorch-vision.")

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import pickle


import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB

logging.getLogger('hpbandster').setLevel(logging.DEBUG)

logging.basicConfig(level=logging.DEBUG)



def get_data():
    # get train data
    train_data_path = '../data/traindata.csv'
    train_label_path = '../data/traindata_label.csv'
    train_x = pd.read_csv(train_data_path)
    train_y = pd.read_csv(train_label_path)

    # get test data
    test_data_path = '../data/testdata.csv'
    test_x = pd.read_csv(test_data_path)

    return train_x, train_y, test_x


def get_combined_data():
    # reading train data
    train_x, train_y,  test_x = get_data()

    combined = train_x.append(test_x)
    combined.reset_index(inplace=True)
    combined.drop(['index'], inplace=True, axis=1)
    return combined, train_y


#Load train and test data into pandas DataFrames
train_x, target ,test_x = get_data()

#Combine train and test data to process them together
combined, target = get_combined_data()

combined['Date'] = pd.to_datetime(combined['Date'], format='%Y-%m-%d')
combined['Date'] = combined['Date'].dt.week

def get_cols_with_no_nans(df, col_type):
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
    else:
        print('Error : choose a type (num, no_num, all)')
        return 0
    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans


num_cols = get_cols_with_no_nans(combined, 'num')
cat_cols = get_cols_with_no_nans(combined, 'no_num')

print('Number of numerical columns with no nan values :', len(num_cols))
print('Number of nun-numerical columns with no nan values :', len(cat_cols))


def oneHotEncode(df, colNames):
    for col in colNames:
        if(df[col].dtype == np.dtype('object')):
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)

            # drop the encoded column
            df.drop([col], axis=1, inplace=True)
    return df


print('There were {} columns before encoding categorical features'.format(
    combined.shape[1]))
combined = oneHotEncode(combined, cat_cols)
print('There are {} columns after encoding categorical features'.format(
    combined.shape[1]))


def split_combined():
    global combined
    train = combined[:16325]
    test = combined[16325:]

    return train, test


train, test = split_combined()


class KerasWorker(Worker):
    def __init__(self, N_train=8192, N_valid=1024, **kwargs):
        super().__init__(**kwargs)

        self.batch_size = 64

        # the data, split between train and test sets
        #(x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test, y_train, y_test = train_test_split(
        train, target, test_size=0.25, random_state=14)

        #x_train = x_train.astype('float32')
        #x_test = x_test.astype('float32')

        # zero-one normalization
        #x_train /= 255
        #x_test /= 255

        # convert class vectors to binary class matrices
        #y_train = keras.utils.to_categorical(y_train, self.num_classes)
        #y_test = keras.utils.to_categorical(y_test, self.num_classes)

        self.x_train, self.y_train = x_train.to_numpy(), y_train.to_numpy()
        self.x_validation, self.y_validation = x_test.to_numpy(), y_test.to_numpy()
        self.x_test, self.y_test = x_test.to_numpy(), y_test.to_numpy()
        self.input_shape = x_train.shape[1]

    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST dataset.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """

        model = Sequential()

        # The Input Layer :
        model.add(Dense(config['num_filters_1'], kernel_initializer='normal',
                    input_dim=self.input_shape, activation='relu'))

        # The Hidden Layers :
        model.add(Dense(config['num_filters_2'],
                    kernel_initializer='normal', activation='relu'))

        # The Output Layer :
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))

        if config['optimizer'] == 'Adam':
            optimizer = tf.keras.optimizers.Adam(lr=config['lr'])
        else:
            optimizer = tf.keras.optimizers.SGD(
                lr=config['lr'], momentum=config['sgd_momentum'])

        model.compile(loss=tf.keras.losses.mean_squared_error,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=int(budget),
                  verbose=0,
                  validation_data=(self.x_test, self.y_test))

        train_score = model.evaluate(self.x_train, self.y_train, verbose=0)
        val_score = model.evaluate(
            self.x_validation, self.y_validation, verbose=0)
        test_score = model.evaluate(self.x_test, self.y_test, verbose=0)

        #import IPython; IPython.embed()
        return ({
                'loss': 1-val_score[1],  # remember: HpBandSter always minimizes!
                'info': {	'test accuracy': test_score[1],
                            'train accuracy': train_score[1],
                            'validation accuracy': val_score[1],
                            'number of parameters': model.count_params()
                        }

                })

    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()

        lr = CSH.UniformFloatHyperparameter(
            'lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)

        # For demonstration purposes, we add different optimizers as categorical hyperparameters.
        # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
        # SGD has a different parameter 'momentum'.
        optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])

        sgd_momentum = CSH.UniformFloatHyperparameter(
            'sgd_momentum', lower=0.0, upper=0.99, default_value=0.9, log=False)

        cs.add_hyperparameters([lr, optimizer, sgd_momentum])

        num_filters_1 = CSH.UniformIntegerHyperparameter(
            'num_filters_1', lower=4, upper=64, default_value=16, log=True)
        num_filters_2 = CSH.UniformIntegerHyperparameter(
            'num_filters_2', lower=4, upper=64, default_value=16, log=True)

        cs.add_hyperparameters([num_filters_1, num_filters_2])

        # The hyperparameter sgd_momentum will be used,if the configuration
        # contains 'SGD' as optimizer.
        cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        cs.add_condition(cond)

        return cs


if __name__ == "__main__":
    working_dir = os.curdir
    # minimum budget that BOHB uses
    min_budget = 1
    # largest budget BOHB will use
    max_budget = 2
    worker = KerasWorker(run_id='0')

    result_file = os.path.join(working_dir, 'bohb_result.pkl')
    nic_name = 'lo'
    port = 0
    run_id = 'bohb_run_1'
    n_bohb_iterations = 1
    
    
    try:
        # Start a nameserver
        host = hpns.nic_name_to_host(nic_name)
        ns = hpns.NameServer(run_id=run_id, host=host, port=port,
                            working_directory=working_dir)
        ns_host, ns_port = ns.start()

        # Start local worker
        w = KerasWorker(run_id=run_id, host=host, nameserver=ns_host,
                        nameserver_port=ns_port, timeout=120)
        w.run(background=True)

        # Run an optimizer
        bohb = BOHB(configspace=worker.get_configspace(),
                    run_id=run_id,
                    host=host,
                    nameserver=ns_host,
                    nameserver_port=ns_port,
                    min_budget=min_budget, max_budget=max_budget)
        
        result = bohb.run(n_iterations=n_bohb_iterations)
        print("Write result to file {}".format(result_file))
        with open(result_file, 'wb') as f:
            pickle.dump(result, f)
    finally:
        bohb.shutdown(shutdown_workers=True)
        ns.shutdown()

    # worker = KerasWorker(run_id='0')
    # cs = worker.get_configspace()

    # config = cs.sample_configuration().get_dictionary()
    # print(config)
    # res = worker.compute(config=config, budget=1000, working_directory='.')
    # print(res)


# NN_model = Sequential()

# # The Input Layer :
# NN_model.add(Dense(12, kernel_initializer='normal',input_dim = train.shape[1], activation='relu'))

# # The Hidden Layers :
# NN_model.add(Dense(8, kernel_initializer='normal',activation='relu'))

# # The Output Layer :
# NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# # Compile the network :
# NN_model.compile(loss=tf.keras.metrics.mean_squared_error, optimizer='rmsprop', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
# NN_model.summary()


# checkpoint_name = 'Weights.hdf5'
# checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
# callbacks_list = [checkpoint]

# NN_model.fit(train, train_y, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

# # Load wights file of the best model :
# wights_file = 'Weights.hdf5' # choose the best checkpoint
# NN_model.load_weights(wights_file) # load it
# NN_model.compile(loss=tf.keras.metrics.mean_squared_error, optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

# def make_submission(prediction, sub_name):
#   my_submission = pd.DataFrame({'ID':pd.read_csv('../data/testdata.csv').index,'AveragePrice':prediction})
#   my_submission.to_csv('../result/{}'.format(sub_name),index=False)
#   print('A submission file has been made')

# predictions = NN_model.predict(test)
# make_submission(predictions[:,0],'submission(NN).csv')
