########################################
########################################
####### Author : Abhinandan Dubey (alivcor)
####### Stony Brook University


import DeepScore_Word2Vec
import DataPreprocessor
import time
import datetime
from keras import optimizers
from keras import regularizers
import EventIssuer
import csv
import sys
import math
from nltk import word_tokenize
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
from keras.models import model_from_yaml
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
import DeepScore_Metrics
import Analyzer
# fix random seed for reproducibility
import pickle
import keras.backend as K

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

    model.add(LSTM(16, return_sequences=True, activation='tanh'))
    model.add(LSTM(12, return_sequences=True))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_X, train_Y, epochs=200, batch_size=10)

    saveModel(model, _LOGFILENAME, timestamp)
    # res = model.predict(test_X)

    scores = model.evaluate(test_X, test_Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    EventIssuer.issueExit(_LOGFILENAME, timestamp)

def getRMSE(predicted_scores, actual_scores):
    err_val = 0.
    for i in range(0, len(predicted_scores)):
        predicted_score = predicted_scores[i]
        actual_score = actual_scores[i]
        err_val += math.pow((predicted_score - actual_score), 2)
    mse = err_val / len(predicted_scores)
    rmse = math.sqrt(mse)
    return rmse


def testModel(model_fn=None):
    _LOGFILENAME, timestamp = start_deepscore_core()
    EventIssuer.issueMessage("Testing the model", _LOGFILENAME)
    X, Y = loadppData('ppData/X_1492930578.7.ds', 'ppData/Y_1492930578.7.ds')
    test_X = X[1500:1700, :]
    test_Y = Y[1500:1700, ]
    predicted_scores = []
    actual_scores = []

    if(model_fn==None):
        model = loadDeepScoreModel(_LOGFILENAME, "1494040329.92")
    else:
        model = loadDeepScoreModel(_LOGFILENAME, model_fn)
    for essay_vector in test_X:
        predicted_scores.append(np.argmax(np.squeeze(model.predict(essay_vector.reshape(1,-1)))))
    for actvector in test_Y:
        actual_scores.append(np.argmax(actvector))

    predicted_scores = np.array(predicted_scores)
    actual_scores = np.array(actual_scores)

    rmse = getRMSE(predicted_scores, actual_scores)
    print "RMSE is : ", rmse
    # print predicted_scores.shape
    # print actual_scores.shape
    print accuracy_score(actual_scores, predicted_scores)
    print(classification_report(actual_scores, predicted_scores))
    # print predicted_scores
    # print actual_scores
    scores = model.evaluate(test_X, test_Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    EventIssuer.issueSuccess("The essays have been graded. ", _LOGFILENAME, ifBold=True)


    EventIssuer.issueExit(_LOGFILENAME, timestamp)







def traintest_model():
    """
    Experimental Function
    :return: None
    """
    _LOGFILENAME, timestamp = start_deepscore_core()
    EventIssuer.issueMessage("Training a new model", _LOGFILENAME)

    # Load Data
    X, Y = loadppData('ppData/X_1492930578.7.ds', 'ppData/Y_1492930578.7.ds')
    # print X.shape, Y.shape
    # split into input (X) and output (Y) variables

    # Partition into train and test
    train_X, train_Y, dev_X, dev_Y, test_X, test_Y = DataPreprocessor.partitionDataset(X, Y)
    # print "dev_Y[0] :", dev_Y[0]
    EventIssuer.issueMessage("Training Set Size : " + str(train_X.shape[0]) + " | Validation Set Size : " + str(dev_X.shape[0]) + " | Test Set Size : " + str(test_X.shape[0]), _LOGFILENAME)

    # Create Model
    model = Sequential()
    model.add(Dense(12, input_dim=300, activation='tanh', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dense(8, activation='tanh', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dense(13, activation='softmax', kernel_regularizer=regularizers.l2(0.0001)))
    #
    # model = Sequential()
    # model.add(Dense(12, input_dim=300, activation='tanh', kernel_regularizer=regularizers.l2(0.0001)))
    # model.add(Activation('tanh'))
    # model.add(Dense(13, activation='softmax'))

    adam = optimizers.Adam(lr=0.002, epsilon=1e-08)

    # Compile Model
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['mean_absolute_error'])

    # Train
    total_train_time = 0
    total_valid_time = 0

    best_qwk = -1
    argmax_best_qwk = 0
    for epoch_num in range(1000):
        # Training
        start_time = time.time()
        running_model = model.fit(train_X, train_Y, batch_size=50, epochs=1, verbose=0)
        train_time = time.time() - start_time
        total_train_time += train_time

        # Evaluate
        start_time = time.time()

        analyzer_object = Analyzer.AnalyzerObject(model, _LOGFILENAME, dev_X, dev_Y, epoch_num, dev_X.shape[0])
        analyzer_object.analyze()

        if(analyzer_object.qwk > best_qwk):
            best_qwk = analyzer_object.qwk
            best_lwk = analyzer_object.lwk
            argmax_best_qwk = epoch_num
            EventIssuer.issueSuccess("Best QWK seen so far : " + str(best_qwk), _LOGFILENAME, highlight=True)

        valid_time = time.time() - start_time
        total_valid_time += valid_time


        # Issue events
        train_loss = running_model.history['loss'][0]
        train_metric = running_model.history['mean_absolute_error'][0]
        epoch_info_1 = "Epoch " + str(epoch_num) + ", train: " + str(train_time) + "s, validation: " + str(valid_time) + "s"
        epoch_info_2 = "[Train] loss: " + str(train_loss) + ", metric: " + str(train_metric)
        EventIssuer.issueMessage(epoch_info_1, _LOGFILENAME)
        EventIssuer.issueMessage(epoch_info_2, _LOGFILENAME)


    # model.fit(train_X, train_Y, epochs=200, batch_size=10)

    saveModel(model, _LOGFILENAME, timestamp)
    # res = model.predict(test_X)

    scores = model.evaluate(test_X, test_Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    EventIssuer.issueSuccess("Best : QWK = " + str(best_qwk) + " | LWK = " + str(best_lwk) + " at Epoch " + str(argmax_best_qwk), _LOGFILENAME, highlight=True)

    EventIssuer.issueExit(_LOGFILENAME, timestamp)

    # testModel(timestamp)



traintest_model()
# testModel()
# train_LSTM()
# train_model()
#
# start_deepscore_core()
# # preprocessDataset(_LOGFILENAME, timestamp)
# # X, Y = loadppData('ppData/X_' + str(timestamp) + '.ds', 'ppData/Y_' + str(timestamp) + '.ds')
# X, Y = loadppData('ppData/X_1492930578.7.ds', 'ppData/Y_1492930578.7.ds')
#
