########################################
########################################
####### Author : Abhinandan Dubey (alivcor)
####### Stony Brook University


import DeepScore_Word2Vec
import DataPreprocessor
import time
import datetime
import EventIssuer
import csv
import sys
from nltk import word_tokenize
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.models import model_from_yaml
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
# fix random seed for reproducibility
import pickle

_LOGFILENAME = ""
reload(sys)
sys.setdefaultencoding('utf8')

def start_deepscore_core():
    global _LOGFILENAME, timestamp
    timestamp = time.time()
    strstamp = datetime.datetime.fromtimestamp(timestamp).strftime('%m-%d-%Y %H:%M:%S')
    _LOGFILENAME = "logs/DeepScore_Log_" + str(timestamp)
    np.random.seed(7)
    EventIssuer.issueWelcome(_LOGFILENAME)
    EventIssuer.genLogFile(_LOGFILENAME, timestamp, strstamp)
    return _LOGFILENAME, timestamp


def loadppData(xfname, yfname):
    X = pickle.load(open(xfname, "r"))
    Y = pickle.load(open(yfname, "r"))
    return X, Y

def saveModel(model, _LOGFILENAME, timestamp):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open('models/model_' + str(timestamp) + '.dsm', "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights('models/weights_' + str(timestamp) + '.h5')
    EventIssuer.issueSuccess("Saved model to disk", _LOGFILENAME)


def loadDeepScoreModel(_LOGFILENAME, model_fn):
    EventIssuer.issueMessage("Loading DeepScore Model : " + model_fn, _LOGFILENAME)
    # dsm = pickle.load(open('models/model_' + str(model_fn) + '.dsm', 'r'))

    # load YAML and create model
    yaml_file = open('models/model_' + str(model_fn) + '.dsm', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights('models/weights_' + str(model_fn) + '.h5')

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    EventIssuer.issueSuccess("Loaded Model Successfully.", _LOGFILENAME)
    return loaded_model

def train_model():
    _LOGFILENAME, timestamp = start_deepscore_core()
    EventIssuer.issueMessage("Training a new model", _LOGFILENAME)
    X, Y = loadppData('ppData/X_1492930578.7.ds', 'ppData/Y_1492930578.7.ds')
    # print X.shape, Y.shape
    # split into input (X) and output (Y) variables
    train_X = X[0:1500,:]
    train_Y = Y[0:1500,]
    test_X = X[1500:1700,:]
    test_Y = Y[1500:1700,]

    print train_X.shape, train_Y.shape, test_X.shape, test_Y.shape
    model = Sequential()
    model.add(Dense(12, input_dim=300, activation='relu'))
    model.add(Dense(8, activation='tanh'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(13, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_X, train_Y, epochs=200, batch_size=10)

    saveModel(model, _LOGFILENAME, timestamp)
    # res = model.predict(test_X)

    scores = model.evaluate(test_X, test_Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    EventIssuer.issueExit(_LOGFILENAME, timestamp)


def train_LSTM():
    _LOGFILENAME, timestamp = start_deepscore_core()
    EventIssuer.issueMessage("Training a new model", _LOGFILENAME)
    X, Y = loadppData('ppData/X_1492930578.7.ds', 'ppData/Y_1492930578.7.ds')
    # print X.shape, Y.shape
    # split into input (X) and output (Y) variables
    train_X = X[0:1500, :]
    train_Y = Y[0:1500, ]
    test_X = X[1500:1700, :]
    test_Y = Y[1500:1700, ]

    model = Sequential()
    model.add(LSTM(32, input_dim=300, return_sequences=True))
    # now model.output_shape == (None, 32)
    # note: `None` is the batch dimension.

    # for subsequent layers, no need to specify the input size:
    model.add(LSTM(16, return_sequences=True, activation='tanh'))

    # to stack recurrent layers, you must use return_sequences=True
    # on any recurrent layer that feeds into another recurrent layer.
    # note that you only need to specify the input size on the first layer.

    model.add(LSTM(12, return_sequences=True))


    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_X, train_Y, epochs=200, batch_size=10)

    saveModel(model, _LOGFILENAME, timestamp)
    # res = model.predict(test_X)

    scores = model.evaluate(test_X, test_Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    EventIssuer.issueExit(_LOGFILENAME, timestamp)

train_LSTM()
# train_model()
#
# start_deepscore_core()
# # preprocessDataset(_LOGFILENAME, timestamp)
# # X, Y = loadppData('ppData/X_' + str(timestamp) + '.ds', 'ppData/Y_' + str(timestamp) + '.ds')
# X, Y = loadppData('ppData/X_1492930578.7.ds', 'ppData/Y_1492930578.7.ds')
#
