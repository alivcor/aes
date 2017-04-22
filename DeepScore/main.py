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
import pickle

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
        word_tokens = word_tokenize(essay)
        wcount = len(word_tokens)
        for word in word_tokens:
            try:
                wvec = local_cache[word]
                total_hits += 1
                if(total_done % 10 == 0):
                    EventIssuer.issueSuccess("Cache Hit ! Current Hit Rate : " + str(round(total_hits*100/total_tokens_processed, 2)) + "%", _LOGFILENAME)
                    EventIssuer.issueMessage("Saving the dictionary at " + str(total_done), _LOGFILENAME)
                    with open('w2v_dict_' + str(timestamp) + '.ds', 'w') as f:
                        pickle.dump(local_cache, f)
                cache_hit_points.append([total_done, round(total_hits*100/total_tokens_processed, 2)])
            except KeyError:
                wvec = glove.getWordVec(word, logfilename=_LOGFILENAME)
                local_cache[word] = wvec
            if type(wvec) != list:
                essay_vector = np.add(essay_vector, wvec)
            else:
                for wvector in wvec:
                    essay_vector = np.add(essay_vector, wvector)
            total_tokens_processed += 1
        total_done += 1.
        essay_vector = essay_vector/wcount
        X.append(essay_vector)
        Y.append(score)
        EventIssuer.issueSharpAlert("Complete: " + str(round(total_done*100/total_essays, 2)) + "%", _LOGFILENAME)
        if(total_done >= 50):
            EventIssuer.issueWarning("Stopping at 50 essays.", _LOGFILENAME)
            break

afterEnd = time.time()

EventIssuer.issueSuccess(str(int(total_done)) + " essays processed in " + str(afterEnd-beforeStart), _LOGFILENAME, ifBold=True)

with open('w2v_dict.ds', 'w') as f:
    pickle.dump(local_cache, f)

with open('cache_hit_rate.csv', 'w') as f:
    for i in cache_hit_points:
        f.write(str(i[0]) + "," + str(i[1]) + "\n")


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