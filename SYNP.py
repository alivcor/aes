########################################
########################################
####### Author : Abhinandan Dubey (alivcor)
####### Stony Brook University

# perfect essays : 37, 118, 147,
import csv
import sys
import nltk
import numpy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.random_projection import sparse_random_matrix
from scipy import spatial
from nltk import word_tokenize, pos_tag
import re

esstxt = "@ORGANIZATION1, Computers are great tools and a great piece of modern technology. Almost every family has them. About @PERCENT1 of my class has computers. So many people have them because their helpful and another current learning resource. Also it's a gr"

esstxt = re.sub(r'(\@)([A-Za-z]*)([\W]*[\d]*[\W]*)(\s)', " ", esstxt)

word = word_tokenize(esstxt)
pos = pos_tag(word)
print(pos)
grammar = "NP: {<IN>?<IN>?<RB>?<DT>?<JJ>*<NN>}"
grammar = """
	NP:   {<IN>?<IN>?<RB>?<DT>?<PRP>?<JJ.*>*<NN.*>+<IN>?<JJ>?<NN>?}
	CP:   {<JJR|JJS>}
	VP: {<VB.*>}
	COMP: {<DT>?<NP><RB>?<VERB><DT>?<CP><THAN><DT>?<NP>}
	"""
ncount = 0;
vcount = 0;

def extract_ideas(t):
    try:
        t.label
    except AttributeError:
        return
    else:
        if t._label == "NP":
            print "t._label : " + t._label
            print "t[0] : " + str(t[0])
            for child in t:
                print str(child[0])
        if t._label == "VP":
            print "t_label : " + t._label
            print "t[0] : " + str(t[0])
            for child in t:
                print str(child[0])
        for child in t:
            extract_ideas(child)


#TODO : Detect variations in tense.
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
sents = sent_detector.tokenize(esstxt.strip())
for sent in sents:
    words = word_tokenize(sent)
    tagged_words = pos_tag(words)
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(tagged_words)
    print result
    print "\n"
    print type(result)
    print "~~~~~~~~~~"
    extract_ideas(result)
    result.draw()
    break

#(\@)([A-Za-z]*)([\W]*[\d]*[\W]*)(\s)
# pf = open('pos_tags.txt', 'w')
# for i in range(0, len(pos)):
#     pf.write(str(pos[i]) + '\n')
#     # if ((pos[i][1]=='NN') or ( pos[i][1]=='NNS')):
#     #    ncount+=1
#     #    nf.write(pos[i][0]+'\n')
# # print(ncount)
# # for i in range(0,len(pos)):
# #    if (pos[i][1]=='VB'):
# #        vcount+=1
# #        vf.write(pos[i][0]+'\n')
# # nf.close()
# # vf.close()
# pf.close()

