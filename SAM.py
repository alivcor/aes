########################################
########################################
####### Author : Abhinandan Dubey (alivcor)
####### Stony Brook University

import csv
import sys
from nltk import word_tokenize, pos_tag
from string import punctuation
import nltk.data

class Essay:
    'Common base class for all essays'

    def __init__(self, ess_id, ess_set, ess_score_r1, ess_score_r2, wcount, lwcount, scount, avslength):
        self.ess_id = ess_id
        self.ess_set = ess_set
        self.ess_score_r1 = ess_score_r1
        self.ess_score_r2 = ess_score_r2
        self.wcount = wcount
        self.lwcount = lwcount
        self.scount = scount
        self.avslength = avslength
    def displayProfile(self):
        print "ID : ", self.ess_id, ", Set: ", self.ess_set, ", SR1: ", self.ess_score_r1, ", SR2: ", self.ess_score_r2

    def getProfile(self):
        return [self.ess_id, self.ess_set, self.ess_score_r1, self.ess_score_r2, self.wcount, self.lwcount, self.scount, self.avslength]

count = 0
esslist = []
# ess3txt = ""
f = open('Dataset/Set1Complete.csv', 'rb')
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
            contents = word_tokenize(ess_text)
            wcount = 0
            for line in contents:
                if (line != "\n"):
                    if line.rstrip() not in punctuation:
                        wcount = wcount + 1
            lwcount = 0
            for line in contents:
                if (line != "\n"):
                    if (len(line.rstrip()) >= 8):
                        lwcount = lwcount + 1
            sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
            sents = sent_detector.tokenize(ess_text.strip())
            i = 0
            avslength = 0
            for line in sents:
                i+=1
                sen_words = word_tokenize(line)
                swcount = 0
                for wline in sen_words:
                    if (wline != "\n"):
                        if wline.rstrip() not in punctuation:
                            swcount = swcount + 1
                avslength = (avslength*(i-1)+swcount)/i
            scount = len(sents)
            ess = Essay(ess_id, ess_set, ess_score_r1, ess_score_r2, wcount, lwcount, scount, avslength)
            esslist.append(ess)
            # JUST FOR TEST
            # print count
            # if ess_id == 4:
            #     ess3txt = ess_text
        else:
            count+=1
finally:
    f.close()

# print esslist[3].__dict__
sam_results = open('Results/stage1_results.csv', 'w')
for ess in esslist:
    sam_results.write(str(ess.getProfile())[1:-1])
    sam_results.write("\n")
count = count-1