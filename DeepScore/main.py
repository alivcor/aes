import glove
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


def loadppData(xfname, yfname):
    X = pickle.load(open(xfname, "r"))
    Y = pickle.load(open(yfname, "r"))
    return X, Y

def encodeOneHot(score):
    onehot_y = [0] * 13
    onehot_y[int(score)] = 1
    return np.array(onehot_y)

def preprocessDataset():
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
                    wvec = glove.getWordVec(word, logfilename=_LOGFILENAME)
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




start_deepscore_core()
# preprocessDataset()
# X, Y = loadppData('ppData/X_' + str(timestamp) + '.ds', 'ppData/Y_' + str(timestamp) + '.ds')
X, Y = loadppData('ppData/X_1492930578.7.ds', 'ppData/Y_1492930578.7.ds')

print X.shape, Y.shape
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

# with open('models/model_' + str(timestamp) + '.dsm', 'w') as f:
#     pickle.dump(model, f)
# res = model.predict(test_X)

scores = model.evaluate(test_X, test_Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

EventIssuer.issueExit(_LOGFILENAME, timestamp)