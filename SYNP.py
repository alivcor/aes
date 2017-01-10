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
esstxt = "I think the use of computers is very good things in the present day @LOCATION2. I think this because people use it to research, stay in touch with friends and family, and music. These are all things adults and kids need a like to do. read on and I will explain why this is so important. My first reason why I think computers are good thing is because the are a good source to do research on a specific topic. Kids all over the @LOCATION1 use the internet to do class projects. Adults can use the computer to look up a certain food resipe. If you need help with something the computer is allways there. Some people even use the computer for a job. Computers can help with a lot of things. Another reason why computers are a big help because you can stay in touch with friends and family. A great thing to use on a computer is a web cam. With this you can see and talk to people on a lillte camray even if they are far away. You can also go on myspace of face book. Theare two websites that are used to post pictures and to talk to chosen friend. It is a good way for people to get to know you. A similar thing to this is instant messaging. You can use this to talk to friends and many on line. Don't stop reading there is more thing! My last reason why computer are sutch a great thing to have is because you can do do mutch with music on the computer. You can use the computer to listen to music, and put music on to and ipod or @NUM1 player. Some people use the computer to make music. You can get a progrem that you can make heats and put it together to make some kind of a song. music is a great thing that all age groups love. That is why I think haveing a computer is a very positive thing. You can do researching, say in touch with friends and family, and do a lot with music. There is so mutch more then thease @NUM2 thing that a computer is good for. I don't think the world would be the same without computers."

esstxt = re.sub(r'(\@)([A-Za-z]*)([\W]*[\d]*[\W]*)(\s)', " ", esstxt)

word = word_tokenize(esstxt)
pos = pos_tag(word)
print(pos)
grammar = "NP: {<IN>?<IN>?<RB>?<DT>?<JJ>*<NN>}"
grammar = """
	NP:   {<IN>?<IN>?<RB>?<DT>?<PRP>?<JJ.*>*<NN.*>+<IN>?<JJ>?<NN>?<CC>?<NN>?}
	CP:   {<JJR|JJS>}
	VP: {<VB.*>}
	COMP: {<DT>?<NP><RB>?<VERB><DT>?<CP><THAN><DT>?<NP>}
	"""
ncount = 0;
vcount = 0;

global ideas_np
global ideas_vp

ideas_np = []
ideas_vp = []

def extract_ideas(t, inp, ivp):
    try:
        t.label
    except AttributeError:
        return
    else:
        if t._label == "NP":
            # print "t._label : " + t._label
            # print "t[0] : " + str(t[0])
            temp = []
            for child in t:
                npw_ = str(child[0])
                print "npw_" + npw_
                #TODO : HERE, ADD ONLY Nouns and adjective
                if child[0] == "NP" or child[0] == "JJ" or child._label == "NNS":
                    temp.append(npw_)
            inp.append(temp)
        if t._label == "VP":
            # print "t_label : " + t._label
            # print "t[0] : " + str(t[0])
            temp = []
            for child in t:
                vpw_ = str(child[0])
                # print vpw_
                temp.append(vpw_)
            ivp.append(temp)
        for child in t:
            extract_ideas(child, inp, ivp)
    return [inp, ivp]


#TODO : Detect variations in tense.
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
sents = sent_detector.tokenize(esstxt.strip())

for sent in sents:
    words = word_tokenize(sent)
    tagged_words = pos_tag(words)
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(tagged_words)

    # print result
    # print "\n"
    # print type(result)
    # print "~~~~~~~~~~"
    inp = []
    ivp = []
    inp, ivp = extract_ideas(result, inp, ivp)
    ideas_np.append(inp)
    ideas_vp.append(ivp)
    # result.draw()

print ideas_np
print ideas_vp


print "Author presents the following key ideas: \n"

for nps in ideas_np:
    for nptuples in nps:
        print "-",
        for wnps in nptuples:
            print wnps,
    print "\n"


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

