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
vectorizer = TfidfVectorizer(max_features=10000,
                             min_df=0.5, stop_words='english',
                             use_idf=True)

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
        return [self.ess_id, self.ess_set, self.ess_score_r1, self.ess_score_r2, self.wcount, self.lwcount, self.scount, self.pcncount, self.avslength]

f1 = open('Results/stage1_results.csv', 'rb')

count = 0
esslist = []
# ess3txt = ""
f = open('Dataset/Set1Complete.csv', 'rb')
resslist = []
esstxts = []
term_freq_matrix=[]
esstxts.append("@ORGANIZATION1, Computers are great tools and a great piece of modern technology. Almost every family has them. About @PERCENT1 of my class has computers. So many people have them because their helpful and another current learning resource. Also it's a gr")
esstxts.append("I go out on sunday nights to play football. We all like sports. Hockey is also a sport. Badminton is also one of my favorite sports")
esstxts.append("Do you love to cook? Cooking Channel's shows and top global chefs share their best recipes and demonstrate their specialties in cooking technique videos.")
esstxts.append("I like computers. Computers provide a lot of help to the users through effective use of technology. Email and messaging and social media are enabled only because of computers")

try:
    reader = csv.reader(f)
    for row in reader:
        if count > 0:
            count+=1
            ess_id = int(row[0])
            ess_set = int(row[1])
            ess_text = unicode(row[2], errors='ignore')
            ess_score_r1 = float(row[3])
            ess_score_r2 = float(row[4])
            if (ess_score_r1+ess_score_r2)==12:
                resslist.append(ess_text)
                ess = Essay(ess_id, ess_set, ess_text, ess_score_r1, ess_score_r2)
                esslist.append(ess)
            elif ess_id ==447:
                esstxts.append(ess_text)
        else:
            count+=1
finally:
    f.close()
#470 447
esstxts.append(" ".join(resslist))

ignorechars = ''',:'!@'''

print len(esslist)
X = vectorizer.fit_transform(esstxts)

tfidf =  X.toarray()
idf = vectorizer.idf_
print tfidf.shape
print tfidf

# csim = 1 - spatial.distance.cosine(tfidf[0], tfidf[5])
# print "CS 0/5", csim
# csim = 1 - spatial.distance.cosine(tfidf[1], tfidf[5])
# print "CS 1/5", csim
# csim = 1 - spatial.distance.cosine(tfidf[2], tfidf[5])
# print "CS 2/5", csim
# csim = 1 - spatial.distance.cosine(tfidf[3], tfidf[5])
# print "CS 3/5", csim
# csim = 1 - spatial.distance.cosine(tfidf[4], tfidf[5])
# print "CS 4/5", csim
# csim = 1 - spatial.distance.cosine(tfidf[5], tfidf[5])
# print "CS 5/5", csim
# #
print dict(zip(vectorizer.get_feature_names(), idf))

U, s, V = np.linalg.svd(tfidf, full_matrices=True)
print U.shape, s.shape, V.shape
#
print "\n\n\n"

print U
# print "~~~~~~~~~~~~~~\n"
# print s
# print "~~~~~~~~~~~~~~\n"
# print V


svd = TruncatedSVD(n_iter=7, random_state=42, n_components=100)
svd.fit(tfidf)
#
#
print("svd.explained_variance_ratio_" + svd.explained_variance_ratio_)

print(svd.explained_variance_ratio_.sum())



