import glove
import time
import datetime
import EventIssuer
import csv
from nltk import word_tokenize
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
# fix random seed for reproducibility


_LOGFILENAME = ""

def start_deepscore_core():
    global _LOGFILENAME, timestamp
    timestamp = time.time()
    strstamp = datetime.datetime.fromtimestamp(timestamp).strftime('%m-%d-%Y %H:%M:%S')
    _LOGFILENAME = "logs/DeepScore_Log_" + str(timestamp)
    np.random.seed(7)
    EventIssuer.issueWelcome(_LOGFILENAME)
    EventIssuer.genLogFile(_LOGFILENAME, timestamp, strstamp)

local_cache = {}
start_deepscore_core()
# print glove.getWordVec("hello", _LOGFILENAME)
total_hits = 0.
total_done = 0.

with open("../Dataset/Set1Complete.csv", "rb") as csvfile:
    datareader = csv.reader(csvfile)
    next(datareader, None)
    X = []
    Y = []
    total_essays = len(datareader)
    for row in datareader:
        essay_vector = np.zeros(300,)
        essay = row[2]
        score = float(float(row[3])+float(row[4]))
        word_tokens = word_tokenize(essay)
        wcount = len(word_tokens)
        for word in word_tokens:
            try:
                wvec = local_cache[word]
                total_hits += 1
                EventIssuer.issueSuccess("Cache Hit ! Current Hit Rate : " + str(total_hits*100/total_done) + "%", _LOGFILENAME)
            except KeyError:
                wvec = glove.getWordVec(word, logfilename=_LOGFILENAME)
                local_cache[word] = wvec
            if type(wvec) != list:
                essay_vector = np.add(essay_vector, wvec)
            else:
                for wvector in wvec:
                    essay_vector = np.add(essay_vector, wvector)
            total_done += 1
        essay_vector = essay_vector/wcount
        X.append(essay_vector)
        Y.append(score)
        EventIssuer.issueSharpAlert("Complete: " + str(total_done*100/total_essays) + "%", _LOGFILENAME)


print X.shape, Y.shape
# split into input (X) and output (Y) variables
# X = dataset[:,2:3]
# Y = dataset[:,3]


# model = Sequential()
# model.add(Dense(12, input_dim=8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# model.fit(X, Y, epochs=150, batch_size=10)
#
# scores = model.evaluate(X, Y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

EventIssuer.issueExit(_LOGFILENAME, timestamp)