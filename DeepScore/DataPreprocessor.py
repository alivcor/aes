########################################
########################################
####### Author : Abhinandan Dubey (alivcor)
####### Stony Brook University


import DeepScore_Word2Vec
import time
import datetime
import EventIssuer
import csv
import sys
from nltk import word_tokenize
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
# fix random seed for reproducibility
import pickle



def encodeOneHot(score):
    onehot_y = [0] * 13
    onehot_y[int(score)] = 1
    return np.array(onehot_y)


def preprocessEssayText(_LOGFILENAME, essay_filename):
    EventIssuer.issueMessage("Loading local dictionary : w2v_dict_final_1492930578.7.dsd", _LOGFILENAME)
    local_cache = pickle.load(open("dictionaries/w2v_dict_final_1492930578.7.dsd", "r"))
    EventIssuer.issueSuccess("Local dictionary loaded.", _LOGFILENAME)
    total_hits = 0.
    total_tokens_processed = 0.
    X = []
    beforeStart = time.time()
    essay_vector = np.zeros(300, )
    EventIssuer.issueMessage("Reading the Essay file provided to me " + essay_filename, _LOGFILENAME)
    with open(essay_filename, 'r') as f:
        essay = f.read()
    try:
        word_tokens = word_tokenize(essay)
    except UnicodeDecodeError:
        essay = essay.decode('latin-1').encode("utf-8")
        word_tokens = word_tokenize(essay)
    wcount = len(word_tokens)
    EventIssuer.issueSleep("Processing the file now.", _LOGFILENAME)
    for word in word_tokens:
        total_tokens_processed += 1
        try:
            wvec = local_cache[word]
            total_hits += 1
        except KeyError:
            wvec = DeepScore_Word2Vec.getWordVec(word, logfilename=_LOGFILENAME)
            local_cache[word] = wvec
        if type(wvec) != list:
            essay_vector = np.add(essay_vector, wvec)
        else:
            for wvector in wvec:
                essay_vector = np.add(essay_vector, wvector)
    essay_vector = essay_vector / wcount
    afterEnd = time.time()
    EventIssuer.issueSuccess("Preprocessed the essay in " + str(afterEnd-beforeStart) + " | Hit Rate : " + str(total_hits*100/total_tokens_processed), _LOGFILENAME)
    X.append(essay_vector)
    return np.array(X)

def preprocessDataset(_LOGFILENAME, timestamp):
    # local_cache = pickle.load(open("dictionaries/w2v_dict_1492912478.79.dsd", "r" ))
    # print glove.getWordVec("hello", _LOGFILENAME)
    local_cache = {}
    total_hits = 0.
    total_done = 0.
    total_tokens_processed = 0.
    cache_hit_points = []

    beforeStart = time.time()
    with open("../Dataset/Set1Complete.csv", "rb") as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader, None)
        X = []
        Y = []
        total_essays = 1783.
        for row in datareader:
            essay_vector = np.zeros(300,)
            essay = row[2]
            score = float(float(row[3])+float(row[4]))
            try:
                word_tokens = word_tokenize(essay)
            except UnicodeDecodeError:
                essay = essay.decode('latin-1').encode("utf-8")
                word_tokens = word_tokenize(essay)
            wcount = len(word_tokens)
            for word in word_tokens:
                total_tokens_processed += 1
                try:
                    wvec = local_cache[word]
                    total_hits += 1
                    cache_hit_points.append([total_done, round(total_hits*100/total_tokens_processed, 2)])
                except KeyError:
                    wvec = DeepScore_Word2Vec.getWordVec(word, logfilename=_LOGFILENAME)
                    local_cache[word] = wvec
                if type(wvec) != list:
                    essay_vector = np.add(essay_vector, wvec)
                else:
                    for wvector in wvec:
                        essay_vector = np.add(essay_vector, wvector)
            total_done += 1.
            essay_vector = essay_vector/wcount
            X.append(essay_vector)
            Y.append(encodeOneHot(score))
            EventIssuer.issueSharpAlert("Complete: " + str(round(total_done*100/total_essays, 2)) + "%", _LOGFILENAME)
            if (total_done % 10 == 0):
                EventIssuer.issueSuccess(
                    "Cache Hit ! Current Hit Rate : " + str(round(total_hits * 100 / total_tokens_processed, 2)) + "%",
                    _LOGFILENAME)
                EventIssuer.issueMessage("Saving the dictionary at " + str(total_done), _LOGFILENAME)
                with open('dictionaries/w2v_dict_' + str(timestamp) + '.dsd', 'w') as f:
                    pickle.dump(local_cache, f)
            # if(total_done >= 700):
            #     EventIssuer.issueWarning("Stopping at 700 essays.", _LOGFILENAME)
            #     break

    afterEnd = time.time()
    with open('dictionaries/w2v_dict_final_' + str(timestamp) + '.dsd', 'w') as f:
        pickle.dump(local_cache, f)
    EventIssuer.issueSuccess(str(int(total_done)) + " essays processed in " + str(afterEnd-beforeStart), _LOGFILENAME, ifBold=True)

    X = np.array(X)
    Y = np.array(Y)

    with open('ppData/X_' + str(timestamp) + '.ds', 'w') as f:
        pickle.dump(X, f)

    with open('ppData/Y_' + str(timestamp) + '.ds', 'w') as f:
        pickle.dump(Y, f)

    with open('optimizations/cache_hit_rate' + str(timestamp) + '.csv', 'w') as f:
        for i in cache_hit_points:
            f.write(str(i[0]) + "," + str(i[1]) + "\n")



def randPartition(alldata_X, alldata_Y, _FRACTION):
    """
    adopted from https://gist.github.com/alivcor/9516927cc211c4cf274167d84574e068 by Abhinandan Dubey (alivcor)
    alldata_X : All of your X (Features) data
    alldata_Y : All of your Y (Prediction) data
    _FRACTION : The fraction of data rows you want for train (0.75 means you need 75% of your data as train and 25% as test)
    """
    np.random.seed(0)
    indices = np.arange(alldata_X.shape[0])
    np.random.shuffle(indices)

    dataX = alldata_X[indices]
    dataY = alldata_Y[indices]

    partition_index = int(dataX.shape[0] * _FRACTION)

    trainX = dataX[0:partition_index]
    testX = dataX[partition_index:dataX.shape[0]]

    trainY = dataY[0:partition_index]
    testY = dataY[partition_index:dataY.shape[0]]

    return [trainX, trainY, testX, testY]


def partitionDataset(X, Y):
    trainX, trainY, testX, testY = randPartition(X, Y, 0.80)
    trainX, trainY, devX, devY = randPartition(trainX, trainY, 0.80)
    return trainX, trainY, devX, devY, testX, testY