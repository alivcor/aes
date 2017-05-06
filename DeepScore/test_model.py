import DeepScore_Word2Vec
import DataPreprocessor
import DeepScore_Core
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
import sys, getopt

_LOGFILENAME = ""
reload(sys)
sys.setdefaultencoding('utf8')



def main(argv, _LOGFILENAME, timestamp):
    try:
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
        print 'test_model.py -i <inputfile>'
        EventIssuer.issueExit(_LOGFILENAME, timestamp)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test_model.py -i <inputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            EssayFileName = arg
    essay_vector = DataPreprocessor.preprocessEssayText(_LOGFILENAME, EssayFileName)
    # print essay_vector.shape
    model = DeepScore_Core.loadDeepScoreModel(_LOGFILENAME, "1494040329.92")
    predicted_score = np.argmax(np.squeeze(model.predict(essay_vector)))
    # print predicted_score
    EventIssuer.issueSuccess("The essay has been graded. I think the score should be " + str(predicted_score) + " out of 12", _LOGFILENAME, ifBold=True)

    EventIssuer.issueExit(_LOGFILENAME, timestamp)



if __name__ == "__main__":
    _LOGFILENAME, timestamp = DeepScore_Core.start_deepscore_core()
    main(sys.argv[1:], _LOGFILENAME, timestamp)