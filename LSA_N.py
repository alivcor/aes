########################################
########################################
####### Author : Abhinandan Dubey (alivcor)
####### Stony Brook University

# perfect essays : 37, 118, 147,
import csv
import sys
from nltk.corpus import stopwords
import numpy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from scipy import spatial

transformer = TfidfTransformer(smooth_idf=False)



class LSA(object):
    def __init__(self, stopwords, ignorechars):
        self.stopwords = stopwords
        self.ignorechars = ignorechars
        self.wdict = {}
        self.dcount = 0


class Essay:
    'Common base class for all essays'

    def __init__(self, ess_id, ess_set, ess_text, ess_score_r1, ess_score_r2):
        self.ess_id = ess_id
        self.ess_set = ess_set
        self.ess_text = ess_text
        self.ess_score_r1 = ess_score_r1
        self.ess_score_r2 = ess_score_r2

    def displayProfile(self):
        print "ID : ", self.ess_id, ", Set: ", self.ess_set, ", SR1: ", self.ess_score_r1, ", SR2: ", self.ess_score_r2

    def getProfile(self):
        return [self.ess_id, self.ess_set, self.ess_score_r1, self.ess_score_r2, self.wcount, self.lwcount, self.scount,
                self.pcncount, self.avslength]


count = 0
esslist = []
# ess3txt = ""
f = open('Dataset/Set1Complete.csv', 'rb')
resslist = []
esstxts = []
term_freq_matrix = []

print "Computing the baseline comparison textual content..."
try:
    reader = csv.reader(f)
    for row in reader:
        if count > 0:
            count += 1
            ess_id = int(row[0])
            ess_set = int(row[1])
            ess_text = unicode(row[2], errors='ignore')
            ess_score_r1 = float(row[3])
            ess_score_r2 = float(row[4])
            if (ess_score_r1 + ess_score_r2) == 12:
                resslist.append(ess_text)
                ess = Essay(ess_id, ess_set, ess_text, ess_score_r1, ess_score_r2)
                esslist.append(ess)
        else:
            count += 1
finally:
    f.close()
ressay = " ".join(resslist)
ignorechars = ''',:'!@'''


csim_list = []

f = open('Dataset/Set1Complete.csv', 'rb')
count = 0
try:
    reader = csv.reader(f)
    for row in reader:
        if count > 0:
            ess_id = int(row[0])
            ess_set = int(row[1])
            ess_text = unicode(row[2], errors='ignore')
            esstxts = []
            esstxts.append(ressay)
            esstxts.append(ess_text)
            vectorizer = TfidfVectorizer(max_features=10000,
                                         min_df=0.5, stop_words='english',
                                         use_idf=True)
            X = vectorizer.fit_transform(esstxts)
            tfidf = X.toarray()
            csim = 1 - spatial.distance.cosine(tfidf[1], tfidf[0])
            csim_list.append(csim)
            print count, csim
            count += 1
        else:
            count += 1
finally:
    f.close()

print csim_list

print "Document similarities computed. Now saving to file."

f1 = open('Results/stage1_results.csv', 'rb')
f2 = open('Results/stage2_results.csv', 'w')
i=-1
try:
    reader = csv.reader(f1)
    for row in reader:
        i+=1
        f2.write(str(row[0]) + "," + str(row[1]) + "," + str(row[2]) + "," + str(row[3]) + "," + str(row[4]) + "," + str(row[5]) + "," + str(row[6]) + "," + str(row[7]) + ", " + str(csim_list[i]))
        f2.write("\n")
finally:
    f.close()